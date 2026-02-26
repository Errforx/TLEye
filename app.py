import sys

from flask import Flask, render_template, request, Response, jsonify, session
import cv2
import threading
import os
import atexit
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import controller as ct  # Arduino LED control
from controller import traffic_state
import concurrent.futures  # Added for parallel inference
import queue  # For frame queue
import time  # For timing
import numpy as np
from collections import deque  # For frame timing deque
import sounddevice as sd
import tensorflow_hub as hub
import pandas as pd
import json, os, time
from types import SimpleNamespace  # used for constructing tracker args

# reinforcement learning support (PPO - Proximal Policy Optimization)
from rl_agent_ppo import PPOAgent

# --------------------------------------------------
# adaptive system parameters (will be tuned by RL agent)
gamma_value = 1.0                    # image gamma correction (DISABLED - leave at 1.0)
conf_threshold_nonemergency = 0.07   # confidence for general vehicle detection
conf_threshold_emergency = 0.25      # emergency vehicle detection
nms_iou_threshold = 0.45            # NMS IoU threshold passed to YOLO
track_buffer_size = 30              # frames for ByteTrack to keep lost tracks

# Disable gamma as RL action (remove from action space)
DISABLE_GAMMA_CORRECTION = True     # Gamma was causing brightness issues

# global state for RL (now using PPO)
STATE_SIZE = 6   # defined in design spec
# action space expanded: 0=none,1=inc_conf,2=dec_conf,3=inc_gamma,
# 4=dec_gamma,5=inc_iou,6=dec_iou,7=inc_buffer,8=dec_buffer,
# 9=trigger_alert,10=suppress_alert
ACTION_SIZE = 11
rl_agent = PPOAgent(STATE_SIZE, ACTION_SIZE)
# try restore previous policy if available
if os.path.exists("rl_model.pth"):
    try:
        rl_agent.load("rl_model.pth")
        print("🔁 Loaded PPO model weights from rl_model.pth")
    except Exception as e:
        print("⚠️ Failed to load PPO model:", e)

# metrics for monitoring training
rl_step_count = 0
rl_reward_sum = 0.0
rl_action_counts = {i: 0 for i in range(ACTION_SIZE)}
rl_training_interval = 10  # Train every 10 steps (optimize for real-time FPS)

last_rl_state = None
last_rl_action = None
last_alert_time = time.time()
rl_last_alert = False

# Hybrid alert confidence tracking
hybrid_alert_active = False
hybrid_confidence = 0.0  # 0.0-1.0 confidence in alert

# ensure agent persists its weights when the process stops
@atexit.register

def save_rl_model():
    try:
        rl_agent.save("rl_model.pth")
        print("✅ RL model saved to rl_model.pth")
    except Exception as e:
        print("⚠️ failed saving RL model:", e)

# --------------------------------------------------

# make sure ByteTrack repo yolox package is importable at top level
# when the repository is included in the workspace but not installed.
repo_dir = os.path.join(os.path.dirname(__file__), 'ByteTrack')
if os.path.isdir(repo_dir) and repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# --- ByteTrack support (optional) ---
byte_tracker_available = False
vehicle_tracker = None
emergency_tracker = None

try:
    # Attempt to import the official ByteTrack repository code. Users must
    # install it manually, e.g. "pip install git+https://github.com/ifzhang/ByteTrack.git"
    # Try the standard package import first (yolox path used by the repo/pip install),
    # then fall back to the namespaced path some installs may create.
    
    from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

    byte_tracker_available = True
    # create default argument namespace used by BYTETracker
    bt_args = SimpleNamespace(track_thresh=0.1, match_thresh=0.8, track_buffer=30, mot20=False)
    vehicle_tracker = BYTETracker(bt_args, frame_rate=30)
    emergency_tracker = BYTETracker(bt_args, frame_rate=30)
    print("✅ ByteTrack initialized: trackers ready")
except Exception as e:  # ImportError or other
    # leave trackers as None; the rest of the code will fall back to
    # YOLO's built-in tracking behavior.
    reason = str(e)
    # if the failure is specifically missing yolox, give extra guidance
    if "No module named 'yolox'" in reason:
        print("⚠️  ByteTrack package found but its `yolox` submodule is missing.")
        print("    - make sure you installed ByteTrack correctly, e.g. by cloning the repo")
        print("      and running `pip install -e .` from its root, or adding the repo to PYTHONPATH.")
    else:
        print("⚠️  ByteTrack unavailable; continuing without it. Install from https://github.com/ifzhang/ByteTrack", e)

# --- CPU efficiency settings ---
CPU_EFFICIENT = True
if CPU_EFFICIENT:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

# --- Temporal confidence buffer (EMA) ---
TEMPORAL_ALPHA = 0.4  # EMA alpha for smoothing detection confidence
BUFFER_MIN_CONF = 0.2  # Minimum smoothed confidence required to commit detection
temporal_confidence = {}  # tid -> ema score
temporal_last_update = {}  # tid -> timestamp of last update

def update_temporal_confidence(tid, score, now=None):
    """Update EMA for a track id and return smoothed score."""
    if now is None:
        now = time.time()
    prev = temporal_confidence.get(tid, None)
    if prev is None:
        ema = float(score)
    else:
        ema = TEMPORAL_ALPHA * float(score) + (1 - TEMPORAL_ALPHA) * prev
    temporal_confidence[tid] = ema
    temporal_last_update[tid] = now
    return ema

def cleanup_old_temporal(now=None, max_age=30.0):
    if now is None:
        now = time.time()
    stale = [tid for tid, t in temporal_last_update.items() if now - t > max_age]
    for tid in stale:
        temporal_last_update.pop(tid, None)
        temporal_confidence.pop(tid, None)


# Set audio device (try different indices from sd.query_devices())
#sd.default.device = 18  # Realtek Microphone - adjust if needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'super secret key'  # Required for session
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO models
model = YOLO('./yolo11n-seg_openvino_model')  # General vehicle detection (non-emergency)
model1 = YOLO('./evdetect_openvino_model/', task='detect')  # Emergency vehicle detection (e.g., ambulances)

# COCO vehicle classes for non-emergency
vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
ev_classes = [0, 1, 2, 3, 4]  # all beacon classes for emergency vehicles

# Distance estimation parameters
KNOWN_WIDTH = 1.8  # Average vehicle width in meters (adjust based on typical vehicles)
FOCAL_LENGTH = 600  # Camera focal length in pixels (needs calibration for accuracy)
DISTANCE_THRESHOLD = 66  # meters
BOTTOM_DISTANCE = 10  # Assumed distance at bottom of frame (m)
TOP_DISTANCE = 100  # Assumed distance at top of frame (m)

# Tracking for traffic jam
track_history = {}
last_seen = {}
STATIONARY_DISPLACEMENT = 20  # Max pixels moved in STATIONARY_TIME to be considered stationary
STATIONARY_TIME = 15  # seconds
MIN_POINTS = 10  # min points in history for stability

camera_active = False
camera_thread = None
frame_queue = queue.Queue(maxsize=1)  # Queue for latest processed frame

detections_summary = {
    "total_detections": 0,
    "vehicles_detected": 0,
    "emergency_detected": 0  # Added for emergency vehicles
}

siren_detected = False

# Global toggle for dynamic line
dynamic_line_enabled = True

# Global device indices
camera_device = 0

# === FPS OPTIMIZATION (Real-time Performance) ===
TARGET_FPS = 30  # Target 30 FPS for real-time processing
FRAME_SKIP = 0  # Skip frames (0 = process every frame)
USE_INFERENCE_OPTIM = True  # Use optimized inference
INFERENCE_DEVICE = 'cpu'  # Force CPU for stability (GPU if available: 'cuda' or '0')

# === CONFIG for YAMNet ===
YAMNET_MODEL_PATH = r'sounddetection\yamnet-tensorflow2-yamnet-v1'  # Update this to your extracted SavedModel directory (contains saved_model.pb, variables/, etc.)
CLASS_MAP_PATH = r'sounddetection\yamnet-tensorflow2-yamnet-v1\assets\yamnet_class_map.csv'
CONF_THRESH = 0.3  # Lower threshold for instant detection
SAMPLE_RATE = 16000  # YAMNet expects 16kHz
AUDIO_GAIN = 20.0  # Increased amplification for better microphone quality
PRINT_DEBUG = True

# YAMNet siren-related class IDs (from AudioSet)
SIREN_CLASSES = [316, 317, 318, 319,390]  # Siren, Civil defense siren, Police car (siren), Ambulance (siren), Fire engine (siren)

def sound_detection():
    global siren_detected
    # Load local YAMNet SavedModel
    YAMNET_MODEL = hub.load(YAMNET_MODEL_PATH)
    print("SOUND DETECTION loaded from local directory.")

    # Load class map
    class_map = pd.read_csv(CLASS_MAP_PATH, on_bad_lines='skip')
    class_names = class_map.display_name.tolist()

    # === Utility: Resample and prepare audio for YAMNet ===
    def prepare_audio(audio):
        # YAMNet expects mono, float32, normalized [-1,1], 16kHz
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # To mono
        audio = audio * AUDIO_GAIN  # Amplify to improve signal strength
        audio = np.clip(audio, -1, 1)  # Prevent clipping
        amplified_rms = np.sqrt(np.mean(audio**2))
        if total_frames % 50 == 0:  # Print amplified RMS occasionally
            print(f"Amplified RMS: {amplified_rms:.6f}")
        return audio.astype(np.float32)

    # === Calibration (for noise floor) ===
    CALIB_FILE = r"sounddetection\calibration.json"
    def calibrate():
        print("🔧 Calibrating ambient noise — stay silent for 3 seconds...")
        samples = sd.rec(int(SAMPLE_RATE * 5), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        rms = np.sqrt(np.mean(samples**2))
        noise_floor = float(rms * 1.5)
        with open(CALIB_FILE, "w") as f:
            json.dump({"rms_threshold": noise_floor}, f)
        print(f"✅ Calibration complete. RMS threshold = {noise_floor:.6f}")

    if not os.path.exists(CALIB_FILE):
        calibrate()
    with open(CALIB_FILE, "r") as f:
        CALIB = json.load(f)
    RMS_THRESH = CALIB["rms_threshold"]

    # === Run Loop ===
    print("Live siren detection started. Press Ctrl+C to stop.\n")

    total_frames = 0
    skipped_frames = 0

    # --- Adaptive device selection & sample rate fallback ---
    try:
        # Choose device: prefer sd.default.device if set, otherwise pick first input device
        dev = sd.default.device
        if isinstance(dev, (list, tuple)) and dev[0] is not None:
            device_index = dev[0]
        elif isinstance(dev, int):
            device_index = dev
        else:
            # Find first input-capable device
            device_index = None
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    device_index = i
                    break
        if device_index is None:
            raise RuntimeError('No input audio device found')

        # Try candidate sample rates; fall back to device default if none match
        candidate_rates = [SAMPLE_RATE, 48000, 44100, 32000]
        chosen_rate = None
        for sr in candidate_rates:
            try:
                sd.check_input_settings(device=device_index, channels=1, samplerate=sr)
                chosen_rate = sr
                break
            except Exception:
                continue
        if chosen_rate is None:
            devinfo = sd.query_devices(device_index)
            chosen_rate = int(devinfo.get('default_samplerate', SAMPLE_RATE))

        print(f"Using audio device #{device_index} - samplerate={chosen_rate}")

    except Exception as e:
        print("Audio device setup failed:", e)
        # If device setup fails, continue but attempt recording with defaults
        device_index = None
        chosen_rate = SAMPLE_RATE

    try:
        while True:
            # === Capture audio window (1s for faster response) ===
            try:
                audio = sd.rec(int(chosen_rate * 1), samplerate=chosen_rate, channels=1, dtype="float32", device=device_index)
                sd.wait()
                audio = audio.flatten()
            except sd.PortAudioError as pae:
                # Try fallbacks: different rates and channel counts
                print('PortAudioError while recording, trying fallbacks:', pae)
                fallback_success = False
                for sr in [44100, 48000, 32000, 16000]:
                    try:
                        sd.check_input_settings(device=device_index, channels=1, samplerate=sr)
                        audio = sd.rec(int(sr * 1), samplerate=sr, channels=1, dtype="float32", device=device_index)
                        sd.wait()
                        audio = audio.flatten()
                        chosen_rate = sr
                        fallback_success = True
                        print(f"Fallback succeeded with samplerate={sr}")
                        break
                    except Exception:
                        continue
                if not fallback_success:
                    # Last-resort: try using device default samplerate and channel count
                    try:
                        devinfo = sd.query_devices(device_index) if device_index is not None else sd.query_devices()
                        default_sr = int(devinfo.get('default_samplerate', SAMPLE_RATE))
                        chans = min(1, devinfo.get('max_input_channels', 1))
                        audio = sd.rec(int(default_sr * 1), samplerate=default_sr, channels=chans, dtype="float32", device=device_index)
                        sd.wait()
                        audio = audio.flatten()
                        chosen_rate = default_sr
                        print(f"Used device defaults samplerate={default_sr} channels={chans}")
                    except Exception as e2:
                        print('All audio capture attempts failed:', e2)
                        time.sleep(1.0)
                        continue

            rms = np.sqrt(np.mean(audio**2))
            total_frames += 1

            if total_frames % 100 == 0:
                skip_rate = (skipped_frames / total_frames) * 100
                print(f"Frame {total_frames}: Skip rate {skip_rate:.1f}% | samplerate={chosen_rate}")

            audio = prepare_audio(audio)

            # === Predict with YAMNet (handles multiple frames internally) ===
            scores, embeddings, spectrogram = YAMNET_MODEL(audio)
            scores = scores.numpy().max(axis=0)  # Max over frames

            # Check for siren classes
            siren_confs = scores[SIREN_CLASSES]
            max_conf = np.max(siren_confs)
            max_idx = SIREN_CLASSES[np.argmax(siren_confs)]
            label = class_names[max_idx]

            if max_conf >= CONF_THRESH:
                print(f" {label.upper():<15} | Conf={max_conf:.2f}")
                siren_detected = True
            else:
                if PRINT_DEBUG:
                    print(f"🔈 [Low Conf] {label}={max_conf:.2f}")
                siren_detected = False

    except KeyboardInterrupt:
        print("\n🛑 Detection stopped gracefully.")

def letterbox_resize(frame, target_size=(640, 360)):
    """Resize frame to target_size maintaining aspect ratio with padding."""
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def estimate_distance(box_width):
    """Estimate distance based on box width."""
    if box_width == 0:
        return float('inf')
    return (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

def run_model(model_instance, frame, conf, track=False, iou=None):
    """Helper for threaded YOLO inference.

    ``conf`` is confidence threshold and ``iou`` (if provided) becomes the
    NMS IoU parameter.  Tracking is still delegated to YOLO when ByteTrack is
    not installed; otherwise we simply return raw detection results and the
    caller handles tracking separately.
    """
    kwargs = {}
    if iou is not None:
        kwargs['iou'] = iou
    if track and not byte_tracker_available:
        return model_instance.track(frame, conf=conf, verbose=False, max_det=10, persist=True, **kwargs)[0]
    else:
        return model_instance.predict(frame, conf=conf, verbose=False, max_det=10, **kwargs)[0]

def clahe_frame(frame):
    """Apply CLAHE to enhance contrast in low-light conditions."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


def adjust_gamma(frame, gamma: float) -> np.ndarray:
    """Perform gamma correction on an image.

    Incorporates a lookup table for efficiency. A gamma of 1.0 returns the
    original frame.  Values <1 darken the image; >1 brighten it.
    """
    if gamma == 1.0:
        return frame
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def build_state(frame, confidences, bboxes, current_ids):
    """Construct state vector used by the RL agent."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray)) / 255.0
    det_conf = float(np.mean(confidences)) if confidences else 0.0
    bbox_area = float(np.mean([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in bboxes])) if bboxes else 0.0
    # estimate tracking age from history
    now = time.time()
    ages = []
    for tid in current_ids:
        hist = track_history.get(tid, [])
        if hist:
            ages.append(now - hist[0][0])
    tracking_age = float(np.mean(ages)) if ages else 0.0
    detection_variance = float(np.var(confidences)) if confidences else 0.0
    tsl_alert = now - last_alert_time
    return np.array([avg_brightness, det_conf, bbox_area, tracking_age, detection_variance, tsl_alert], dtype=np.float32)


def apply_rl_action(action):
    """Modify global parameters according to an RL action.
    
    NOTE: Actions 9 (alert) and 10 (suppress) are now handled by HYBRID LOGIC only.
    The RL agent's alert decisions are suggestions that are evaluated against sensor data.
    """
    global gamma_value, conf_threshold_nonemergency, nms_iou_threshold
    global track_buffer_size, bt_args
    global rl_last_alert

    if action == 1:  # increase confidence threshold
        conf_threshold_nonemergency = min(conf_threshold_nonemergency + 0.01, 1.0)
    elif action == 2:  # decrease confidence threshold
        conf_threshold_nonemergency = max(conf_threshold_nonemergency - 0.01, 0.0)
    elif action == 3:  # increase gamma (DISABLED - gamma is staying at 1.0)
        pass  # Gamma correction disabled
    elif action == 4:  # decrease gamma (DISABLED - gamma is staying at 1.0)
        pass  # Gamma correction disabled
    elif action == 5:  # increase nms_iou
        nms_iou_threshold = min(nms_iou_threshold + 0.05, 1.0)
    elif action == 6:  # decrease nms_iou
        nms_iou_threshold = max(nms_iou_threshold - 0.05, 0.0)
    elif action == 7:  # increase track buffer size
        track_buffer_size = min(track_buffer_size + 5, 100)
        # update ByteTrack arguments if available
        if 'bt_args' in globals():
            bt_args.track_buffer = track_buffer_size
        if vehicle_tracker is not None:
            vehicle_tracker.buffer_size = track_buffer_size
            vehicle_tracker.max_time_lost = track_buffer_size
        if emergency_tracker is not None:
            emergency_tracker.buffer_size = track_buffer_size
            emergency_tracker.max_time_lost = track_buffer_size
    elif action == 8:  # decrease track buffer size
        track_buffer_size = max(track_buffer_size - 5, 1)
        if 'bt_args' in globals():
            bt_args.track_buffer = track_buffer_size
        if vehicle_tracker is not None:
            vehicle_tracker.buffer_size = track_buffer_size
            vehicle_tracker.max_time_lost = track_buffer_size
        if emergency_tracker is not None:
            emergency_tracker.buffer_size = track_buffer_size
            emergency_tracker.max_time_lost = track_buffer_size
    elif action == 9:  # HYBRID: suggest alert (will be validated by hybrid_alert_logic)
        rl_last_alert = True
    elif action == 10:  # HYBRID: suggest suppress (will be validated by hybrid_alert_logic)
        rl_last_alert = False
    # action == 0 -> do nothing


def apply_hybrid_alert(should_alert: bool, confidence: float, reason: str):
    """Apply the HYBRID ALERT DECISION to LED controller.
    
    Args:
        should_alert: Whether to activate the siren
        confidence: 0.0-1.0 confidence in the decision
        reason: Why this decision was made (for logging)
    """
    global last_alert_time
    
    if should_alert:
        ct.siren_alert(True)
        last_alert_time = time.time()
        if confidence >= 0.8:
            print(f"  🚨 ALERT ACTIVATED ({reason}, conf={confidence:.2f})")
        else:
            print(f"  ⚠️  ALERT SUGGESTED ({reason}, conf={confidence:.2f})")
    else:
        ct.siren_alert(False)
        if reason != "NO_THREAT":
            print(f"  🔇 Alert suppressed ({reason})")


def hybrid_alert_logic(emergency_flag: bool, siren_flag: bool, rl_alert_decision: bool) -> tuple:
    """
    HYBRID LOGIC: Combines user's reliable LED logic with AI decision-making.
    
    Philosophy:
    1. **Ground Truth First**: Sensor data (emergency + siren) is most reliable
    2. **AI as Confidence Boost**: RL agent provides context awareness
    3. **Safety Override**: Never suppress if sensors detect both threats
    
    Returns: (should_alert: bool, confidence: float, reason: str)
    """
    global hybrid_alert_active, hybrid_confidence
    
    # Ground truth: both emergency vehicle AND siren detected
    ground_truth_alert = emergency_flag and siren_flag
    
    # RL agent's suggestion
    rl_wants_alert = rl_alert_decision
    
    # DECISION LOGIC (Hybrid):
    if ground_truth_alert:
        # CASE 1: Sensors say DEFINITE THREAT → ALWAYS ALERT (safety first)
        should_alert = True
        confidence = 1.0  # Maximum confidence
        reason = "GROUND_TRUTH_ALERT"
    
    elif rl_wants_alert and (emergency_flag or siren_flag):
        # CASE 2: RL suggests alert + at least one sensor confirms threat
        should_alert = True
        confidence = 0.8  # High confidence (RL + partial sensor agreement)
        reason = "RL_ASSISTED_ALERT"
    
    elif emergency_flag and not siren_flag:
        # CASE 3: Emergency vehicle detected but no siren
        # → Output: Alert only if RL agrees (low confidence)
        should_alert = rl_wants_alert
        confidence = 0.4  # Low confidence (one sensor only)
        reason = "VEHICLE_ONLY"
    
    elif siren_flag and not emergency_flag:
        # CASE 4: Siren detected but no emergency vehicle
        # → Might be from nearby vehicle or ambulance out of frame
        should_alert = rl_wants_alert
        confidence = 0.3  # Low confidence (one sensor only)
        reason = "SIREN_ONLY"
    
    else:
        # CASE 5: No sensors triggered
        should_alert = False  # NEVER alert on RL alone
        confidence = 0.0
        reason = "NO_THREAT"
    
    hybrid_alert_active = should_alert
    hybrid_confidence = confidence
    
    return should_alert, confidence, reason


def compute_reward(emergency_flag: bool, siren_flag: bool, stationary_count: int) -> float:
    """Reward shaping for RL learning with hybrid logic.

    The RL agent learns to:
    - Suggest alerts when it detects emergency patterns
    - Suppress when no threat is present
    - Complement sensor data with contextual awareness
    
    Rewards are based on agreement with HYBRID DECISION (not pure sensor truth).
    """
    r = 0.0
    global rl_last_alert, hybrid_alert_active, hybrid_confidence
    
    # RL's role: assist the hybrid decision (not override ground truth)
    if hybrid_alert_active:
        # If alert is active, reward RL if it suggested alert
        if rl_last_alert:
            r += 4.0 * hybrid_confidence  # Reward based on confidence
        else:
            r -= 2.0  # Mild penalty (hybrid overrode RL)
    else:
        # If no alert, reward RL if it suggested suppress
        if rl_last_alert:
            r -= 1.0  # Mild penalty (RL wanted to alert)
        else:
            r += 2.0  # Reward for correct suppress
    
    # SECONDARY: Stability bonus
    stability = min(1.5 * float(stationary_count), 10.0)
    r += stability
    return r

# metric for FPS tracking
frame_times = deque(maxlen=30)  # Track last 30 frame times for FPS calculation
last_frame_time = time.time()
frame_count = 0

def detect_objects(frame):
    """Detect non-emergency and emergency vehicles within 30m, draw boxes, update stats, and trigger LED.

    The routine also builds a state vector for the RL agent, lets the agent choose
    an action, applies any parameter updates, computes a reward and trains the
    network on the resulting transition.
    """
    global detections_summary, track_history, last_seen, siren_detected
    global frame_times, last_frame_time, frame_count

    # === FPS OPTIMIZATION ===
    frame_count += 1
    current_time_ms = time.time()
    
    # Calculate current FPS
    if frame_times:
        avg_frame_time = np.mean(list(frame_times))
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        current_fps = 0
    
    # Skip frames if needed to achieve target FPS
    time_since_last = current_time_ms - last_frame_time
    target_frame_time = 1.0 / TARGET_FPS
    if time_since_last < target_frame_time:
        return frame  # Skip processing, return original frame

    frame_times.append(time_since_last)
    last_frame_time = current_time_ms

    # === pre-process ===
    # DISABLED: Gamma correction was causing brightness issues
    # if gamma_value != 1.0:
    #     frame = adjust_gamma(frame, gamma_value)

    # lists that will be used to generate the RL state
    rl_confs = []
    rl_bboxes = []

    # Resize before inference for speed (reduced resolution for optimization)
    resized = cv2.resize(frame, (640, 360))  # Increased resolution for better visibility
    
    # Calculate threshold line position
    height, width = resized.shape[:2]
    if TOP_DISTANCE > BOTTOM_DISTANCE:
        normalized = (DISTANCE_THRESHOLD - BOTTOM_DISTANCE) / (TOP_DISTANCE - BOTTOM_DISTANCE)
        normalized = max(0, min(1, normalized))  # Clamp to [0,1]
        line_y = int(height * (1 - normalized))  # Bottom at 1, top at 0
    else:
        line_y = int(height * 0.8)  # Fallback
    
    current_time = time.time()
    
    # Run models in parallel using threads (cpu-efficient mode reduces workers)
    max_workers = 1 if CPU_EFFICIENT else 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_non_emergency = executor.submit(
            run_model,
            model,
            resized,
            conf_threshold_nonemergency,
            True,
            nms_iou_threshold,
        )
        if siren_detected:
            future_emergency = executor.submit(
                run_model,
                model1,
                resized,
                conf_threshold_emergency,
                False,
                nms_iou_threshold,
            )
        else:
            future_emergency = None

        results = future_non_emergency.result()
        if siren_detected and future_emergency is not None:
            results1 = future_emergency.result()
        else:
            results1 = None

    # gather raw detections for RL state (before any filtering)
    if results is not None:
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            rl_confs.append(float(conf))
            coords = tuple(map(int, box))
            rl_bboxes.append(coords)
    if results1 is not None:
        for box, conf in zip(results1.boxes.xyxy, results1.boxes.conf):
            rl_confs.append(float(conf))
            coords = tuple(map(int, box))
            rl_bboxes.append(coords)

    vehicle_detected = False
    emergency_detected = False
    vehicle_count = 0
    emergency_count = 0

    traffic_state["vehicle_detected"] = False

    current_ids = set()

    # Process non-emergency detections
    if byte_tracker_available:
        # Build detections array for BYTETracker (vehicles only)
        dets = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            cls = int(cls)
            if cls not in vehicle_classes:
                continue
            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) / 2
            if dynamic_line_enabled and center_y < line_y:
                continue
            dets.append([x1, y1, x2, y2, float(conf)])
        dets = np.array(dets) if len(dets) > 0 else np.zeros((0, 5))
        # supply image size info as expected by BYTETracker
        img_info = (height, width)
        online_targets = vehicle_tracker.update(dets, img_info, (height, width))
        for track in online_targets:
            x1, y1, x2, y2 = map(int, track.tlbr)
            center_y = (y1 + y2) / 2
            if dynamic_line_enabled and center_y < line_y:
                continue
            box_width = x2 - x1
            distance = estimate_distance(box_width)
            color = (0, 255, 0)
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            score = float(getattr(track, 'score', 0.0))
            # Smooth confidence by track id
            tid = track.track_id
            smoothed = update_temporal_confidence(tid, score, current_time)
            if smoothed < BUFFER_MIN_CONF:
                # Skip low-smoothed-confidence detections to avoid flicker/noise
                continue
            cv2.putText(resized, f'ID{tid} {smoothed:.2f} ({distance:.1f}m)',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            vehicle_detected = True
            vehicle_count += 1
            # update track history for traffic jam detection
            tid = track.track_id
            current_ids.add(tid)
            center = ((x1 + x2) / 2, center_y)
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((current_time, center))
            while track_history[tid] and track_history[tid][0][0] < current_time - 60:
                track_history[tid].pop(0)
    else:
        # original YOLO tracking code
        if results.boxes.id is not None:
            for i in range(len(results.boxes)):
                box = results.boxes.xyxy[i]
                cls = int(results.boxes.cls[i])
                if cls not in vehicle_classes:
                    continue
                conf = float(results.boxes.conf[i])
                tid = int(results.boxes.id[i])
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) / 2
                if dynamic_line_enabled and center_y < line_y:
                    continue  # Skip vehicles beyond the threshold line if enabled
                box_width = x2 - x1
                distance = estimate_distance(box_width)
                center = ((x1 + x2) / 2, center_y)
                label = model.names[cls]
                color = (0, 255, 0)  # All detected are within threshold
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)  # Thicker box for visibility
                # Smooth per-track confidence
                smoothed = update_temporal_confidence(tid, conf, current_time)
                if smoothed < BUFFER_MIN_CONF:
                    continue
                cv2.putText(resized, f'{label} {smoothed:.2f} ({distance:.1f}m)', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Added back text with smaller font
                vehicle_detected = True
                vehicle_count += 1
                
                # Update track history
                current_ids.add(tid)
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((current_time, center))
                
                # Clean old entries >60s
                while track_history[tid] and track_history[tid][0][0] < current_time - 60:
                    track_history[tid].pop(0)

    # Process emergency detections if siren detected
    if siren_detected and results1 is not None:
        if byte_tracker_available:
            dets1 = []
            for box, cls, conf in zip(results1.boxes.xyxy, results1.boxes.cls, results1.boxes.conf):
                cls = int(cls)
                if cls not in ev_classes:
                    continue
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) / 2
                if dynamic_line_enabled and center_y < line_y:
                    continue
                dets1.append([x1, y1, x2, y2, float(conf)])
            dets1 = np.array(dets1) if len(dets1) > 0 else np.zeros((0, 5))
            online_targets1 = emergency_tracker.update(dets1, (height, width), (height, width))
            for track in online_targets1:
                x1, y1, x2, y2 = map(int, track.tlbr)
                center_y = (y1 + y2) / 2
                if dynamic_line_enabled and center_y < line_y:
                    continue
                box_width = x2 - x1
                distance = estimate_distance(box_width)
                color = (0, 0, 255)
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
                score = float(getattr(track, 'score', 0.0))
                tid = track.track_id
                smoothed = update_temporal_confidence(f"ev_{tid}", score, current_time)
                if smoothed < BUFFER_MIN_CONF:
                    continue
                cv2.putText(resized, f'EV_ID{tid} {smoothed:.2f} ({distance:.1f}m)',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                emergency_detected = True
                emergency_count += 1
        else:
            for box, cls, conf in zip(results1.boxes.xyxy, results1.boxes.cls, results1.boxes.conf):
                cls = int(cls)
                if cls not in ev_classes:  # Classes 0-4 for all beacons
                    continue
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) / 2
                if dynamic_line_enabled and center_y < line_y:
                    continue  # Skip vehicles beyond the threshold line if enabled
                box_width = x2 - x1
                distance = estimate_distance(box_width)
                label = model1.names[cls]
                conf = float(conf)
                color = (0, 0, 255)  # All detected are within threshold
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)  # Thicker box
                cv2.putText(resized, f'{label} {conf:.2f} ({distance:.1f}m)', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Added back text
                emergency_detected = True
                emergency_count += 1

    # Draw threshold line dynamically based on perspective
    height, width = resized.shape[:2]
    # Calculate normalized position for DISTANCE_THRESHOLD
    if TOP_DISTANCE > BOTTOM_DISTANCE:
        normalized = (DISTANCE_THRESHOLD - BOTTOM_DISTANCE) / (TOP_DISTANCE - BOTTOM_DISTANCE)
        normalized = max(0, min(1, normalized))  # Clamp to [0,1]
        line_y = int(height * (1 - normalized))  # Bottom at 1, top at 0
    else:
        line_y = int(height * 0.8)  # Fallback
    if dynamic_line_enabled:
        cv2.line(resized, (0, line_y), (width, line_y), (255, 0, 0), 2)
        cv2.putText(resized, f'Threshold Line', (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Update last_seen
    for tid in current_ids:
        last_seen[tid] = current_time
    
    # Clean old tracks
    to_remove = [tid for tid in list(track_history) if tid not in current_ids and current_time - last_seen.get(tid, 0) > 10]
    for tid in to_remove:
        del track_history[tid]
        if tid in last_seen:
            del last_seen[tid]
    # cleanup temporal confidence entries for stale tracks
    cleanup_old_temporal(now=current_time, max_age=30.0)
    
    # Count stationary non-emergency vehicles
    stationary_count = 0
    for tid in track_history:
        history = track_history[tid]
        if len(history) < MIN_POINTS:
            continue
        recent_history = [h for h in history if h[0] > current_time - STATIONARY_TIME]
        if len(recent_history) < MIN_POINTS:
            continue
        positions = [pos for t, pos in recent_history]
        first_pos = np.array(positions[0])
        last_pos = np.array(positions[-1])
        displacement = np.linalg.norm(last_pos - first_pos)
        if displacement < STATIONARY_DISPLACEMENT:
            stationary_count += 1
    
    if stationary_count >= 5:  # Threshold for traffic jam
        traffic_state["vehicle_detected"] = True
        print(f"Traffic jammed: {stationary_count} stationary vehicles")
    else:
        traffic_state["vehicle_detected"] = False
        print(f"Not jammed: {stationary_count} stationary vehicles")

    # === reinforcement learning step (PPO) ===
    current_state = build_state(resized, rl_confs, rl_bboxes, current_ids)
    action, value_estimate, log_prob = rl_agent.select_action(current_state)
    apply_rl_action(action)  # apply parameter tuning (actions 0-8)
    
    # === HYBRID ALERT LOGIC (combining user's LED logic + RL agent) ===
    rl_alert_suggestion = (action == 9)  # RL suggests alert if action is 9
    should_alert, confidence, reason = hybrid_alert_logic(emergency_detected, siren_detected, rl_alert_suggestion)
    apply_hybrid_alert(should_alert, confidence, reason)  # Apply the HYBRID decision to LED

    # metrics tracking
    global rl_step_count, rl_reward_sum, rl_action_counts, rl_training_interval
    rl_step_count += 1
    rl_action_counts[action] += 1

    next_state = build_state(resized, rl_confs, rl_bboxes, current_ids)
    reward = compute_reward(emergency_detected, siren_detected, stationary_count)
    rl_reward_sum += reward

    # Store transition for PPO (includes value and log_prob)
    rl_agent.store_transition(current_state, action, reward, value_estimate, log_prob, False)
    
    # Update every N steps (PPO does batch updates)
    if rl_step_count % rl_training_interval == 0:
        rl_agent.update(next_state)

    # print status every 100 steps
    if rl_step_count % 100 == 0:
        avg_reward = rl_reward_sum / rl_step_count
        total_steps = rl_agent.total_steps
        print(f"[PPO] steps={rl_step_count} total_steps_trained={total_steps} avg_reward={avg_reward:.2f}")

    # Update stats (only for detections within 30m)
    detections_summary["vehicles_detected"] = vehicle_count
    detections_summary["emergency_detected"] = emergency_count
    detections_summary["total_detections"] += (vehicle_count + emergency_count)

    # Siren alert already handled in sound_detection

    return resized

def camera_processing_loop():
    """Background thread for camera capture and processing."""
    cap = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW)
    # Remove fixed resolution, use native

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_counter = 0
    while camera_active:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        frame_counter += 1
        if frame_counter % 3 != 0:  # Process every 3rd frame
            time.sleep(0.05)
            continue
        frame = letterbox_resize(frame, (640, 360))  # Resize with aspect ratio maintained
        processed = detect_objects(frame)
        try:
            frame_queue.put_nowait(processed)  # Put latest frame, overwrite if full
        except queue.Full:
            pass
    cap.release()

def generate_frames():
    """Stream latest processed frames."""
    while camera_active:
        try:
            frame = frame_queue.get(timeout=0.1)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Balanced quality
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            pass  # Wait for next frame

def detect_file(filepath):
    """Run detection on uploaded video with optimizations."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % 3 != 0:  # Skip 2/3 frames
            continue
        frame = letterbox_resize(frame, (640, 360))
        frame = detect_objects(frame)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def detect_photo(filepath):
    """Run detection on uploaded photo and return annotated frame."""
    frame = cv2.imread(filepath)
    if frame is None:
        return None
    frame = letterbox_resize(frame, (640, 360))
    frame = detect_objects(frame)
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/video_feed')
def video_feed():
    global camera_active, camera_thread
    camera_active = True
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
        camera_thread.start()
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    if not camera_active:
        with frame_queue.mutex:
            frame_queue.queue.clear()
    return ("Camera turned on" if camera_active else "Camera turned off"), 200

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['photo_filepath'] = filepath
    return jsonify({"redirect_url": '/results?source=photo'})

@app.route('/upload_photo', methods=['GET'])
def upload_photo_get():
    filepath = session.get('photo_filepath')
    if not filepath:
        return "No photo", 400
    image = detect_photo(filepath)
    return Response(image, mimetype='image/jpeg')

@app.route('/upload_video_feed', methods=['POST'])
def upload_video_feed():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['video_filepath'] = filepath
    return jsonify({"redirect_url": '/results?source=video'})

@app.route('/upload_video_feed', methods=['GET'])
def upload_video_feed_get():
    filepath = session.get('video_filepath')
    if not filepath:
        return "No video", 400
    return Response(detect_file(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_display/<filename>')
def video_display(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return Response(detect_file(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(detections_summary)

@app.route('/toggle_dynamic_line', methods=['POST'])
def toggle_dynamic_line():
    global dynamic_line_enabled
    dynamic_line_enabled = not dynamic_line_enabled
    session['dynamic_line_enabled'] = dynamic_line_enabled  # Also update session for persistence
    return "Dynamic line " + ("enabled" if dynamic_line_enabled else "disabled"), 200

@app.route('/get_devices')
def get_devices():
    import cv2
    cameras = []
    for i in range(10):  # Check first 10
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    mics = []
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            mics.append({'index': i, 'name': dev['name']})
    return jsonify({'cameras': cameras, 'mics': mics})

@app.route('/set_devices', methods=['POST'])
def set_devices():
    global camera_device
    data = request.json
    camera_device = int(data.get('camera', 0))
    mic_device = int(data.get('mic', 14))
    sd.default.device = mic_device
    return "Devices set", 200

if __name__ == '__main__':
    sound_thread = threading.Thread(target=sound_detection, daemon=True)
    sound_thread.start()

    ct.start_traffic_control()
    app.run(host='0.0.0.0', port=443, ssl_context=('TLEye.pem', 'TLEye-key.pem'), debug=False)