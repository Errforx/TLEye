from flask import Flask, render_template, request, Response, jsonify, send_from_directory, redirect, url_for
import cv2
import threading
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time
import controller as ct
import io
import base64
from sound_detector import classify_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detection = False

# ==========================================
# Model setup
# ==========================================
model = YOLO('./yolo11n-seg_openvino_model/')
model1 = YOLO('./best_openvino_model/', task='segment')

allowed_model = {1, 2, 3, 5, 7}   # COCO classes
allowed_model1 = {0}               # ambulance class
DEFAULT_CONF_THRESHOLD = 0.20
AMBULANCE_CONF_THRESHOLD = 0.9

camera_active = False

KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 700
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_frame = None
prev_points = None
_no_new_points_age = 0

infer_lock = threading.Lock()
last_results = None
last_results1 = None
_latest_frame_for_inference = None
_inference_worker_stop = False

detections_summary = {"total_detections": 0, "vehicles_detected": 0}


# ==========================================
# Utility and Detection Functions
# ==========================================
def estimate_distance(box):
    box_width = box[2] - box[0]
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
    return distance


def is_near(distance, threshold=4.0):
    return distance < threshold


def track_objects(frame, detections):
    global prev_frame, prev_points, _no_new_points_age
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray_frame
    if prev_frame.shape != gray_frame.shape:
        gray_frame = cv2.resize(gray_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    new_points = []
    if detections:
        for item in detections:
            if item is None:
                continue
            if isinstance(item, tuple):
                res, allowed_set = item
            else:
                res, allowed_set = item, None
            if res is None:
                continue
            for result in getattr(res, "boxes", []):
                box = result.xyxy[0].cpu().numpy()
                cls = int(result.cls[0].cpu().numpy())
                if allowed_set is None or cls in allowed_set:
                    new_points.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    if new_points:
        prev_points = np.float32(new_points).reshape(-1, 1, 2)
        _no_new_points_age = 0
    else:
        _no_new_points_age += 1
        if _no_new_points_age > 3:
            prev_points = None

    try:
        if prev_points is not None and prev_points.size > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_points, None, **lk_params)
            for i, (new, old) in enumerate(zip(next_points, prev_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            prev_points = next_points
    except cv2.error:
        prev_points = None
    prev_frame = gray_frame
    return frame


def detect_and_alert(frame, res=None, res1=None):
    global detection
    try:
        h, w = frame.shape[:2]
        # Primary model (vehicles)
        if res is not None and getattr(res, "boxes", None) is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy().astype(float)
            for b, cid, cf in zip(xyxy, cls, conf):
                if int(cid) not in allowed_model or float(cf) < DEFAULT_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"{model.names.get(int(cid), str(cid))} {cf:.2f}"
                cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                detection = True
        # Secondary model (ambulance)
        if res1 is not None and getattr(res1, "boxes", None) is not None:
            xyxy1 = res1.boxes.xyxy.cpu().numpy()
            cls1 = res1.boxes.cls.cpu().numpy().astype(int)
            conf1 = res1.boxes.conf.cpu().numpy().astype(float)
            for b, cid, cf in zip(xyxy1, cls1, conf1):
                if int(cid) not in allowed_model1 or float(cf) < AMBULANCE_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 4)
                label = f"{model1.names.get(int(cid), str(cid))} {cf:.2f}"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                detection = True
        if detection == True:
           ct.led('car')
           detection = False
        else:
            ct.led('none')
    except Exception as e:
        print("Detect error:", e)
    return frame


# ==========================================
# Real-time Inference Worker and Stream
# ==========================================
def _inference_worker():
    global last_results, last_results1, _latest_frame_for_inference, _inference_worker_stop
    model1_skip = 1
    loop_idx = 0

    while not _inference_worker_stop:
        if _latest_frame_for_inference is None:
            time.sleep(0.005)
            continue
        frame = _latest_frame_for_inference
        _latest_frame_for_inference = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with infer_lock:
                last_results = model.predict(rgb, conf=DEFAULT_CONF_THRESHOLD)[0]
                if (loop_idx % model1_skip) == 0:
                    last_results1 = model1.predict(rgb, conf=AMBULANCE_CONF_THRESHOLD)[0]
            loop_idx += 1
        except Exception as e:
            print("Inference error:", e)
            time.sleep(0.01)


def generate_frames():
    """
    Continuously capture frames, annotate, and yield MJPEG chunks.
    """
    global camera_active, _latest_frame_for_inference
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Camera stream started.")
    while True:
        if not camera_active:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            continue

        _latest_frame_for_inference = frame.copy()

        annotated = detect_and_alert(frame.copy(), last_results, last_results1)
        annotated = track_objects(annotated, [(last_results, allowed_model), (last_results1, allowed_model1)])

        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Critical: yield must use CRLF boundaries exactly like below
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # tiny sleep avoids CPU max-out but still real-time
        time.sleep(0.01)



threading.Thread(target=_inference_worker, daemon=True).start()


# ==========================================
# Flask Routes
# ==========================================
@app.route('/')
def index():
    preview = request.args.get('preview')
    msg = request.args.get('msg')
    return render_template('index.html', preview_url=preview, upload_message=msg,
                           camera_active=camera_active, server_response=None)
@app.route('/toggle_camera')
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'active': camera_active})


# === API route for siren detection ===
@app.route("/api/audio", methods=["POST"])
def api_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    try:
        result = classify_audio(save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except:
            pass

@app.route('/video_feed')
def video_feed():
    """
    Stream the live annotated camera feed as MJPEG.
    """
    global camera_active
    camera_active = True  # make sure stream loop runs

    # Important: use "multipart/x-mixed-replace"
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        direct_passthrough=True  # allow continuous flush
    )


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    upload_path = os.path.join('uploads', secure_filename(file.filename))
    os.makedirs('uploads', exist_ok=True)
    file.save(upload_path)

    ext = file.filename.lower().split('.')[-1]
    output_name = 'output_' + file.filename
    out_path = os.path.join('static', output_name)

    # Process image file
    if ext in ['jpg', 'jpeg', 'png']:
        frame = cv2.imread(upload_path)

        # Run both models
        res = model(frame)
        res1 = modell(frame)

        # Use your existing detector
        detect_and_alert(frame, res, res1)

        # Save annotated output
        cv2.imwrite(out_path, frame)
        return jsonify({'image_url': url_for('static', filename=output_name)})

    # Process video file
    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        cap = cv2.VideoCapture(upload_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run both models
            res = model(frame)
            res1 = modell(frame)

            # Annotate using your detector
            detect_and_alert(frame, res, res1)
            out.write(frame)

        cap.release()
        out.release()

        return jsonify({'video_url': url_for('static', filename=output_name)})

    else:
        return jsonify({'error': 'Unsupported file format'}), 400



@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/stats')
def stats():
    global detections_summary
    return jsonify(detections_summary)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
