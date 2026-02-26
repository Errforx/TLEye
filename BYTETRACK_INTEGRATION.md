# ByteTrack Integration Guide

This project now includes optional support for the [ByteTrack](https://github.com/ifzhang/ByteTrack) multi-object tracker.  When ByteTrack is available the application will use it instead of the built-in YOLOv8 tracker for more robust ID assignment and fewer missed/fragmented tracks.

## Installation steps
1. **Clone or install the ByteTrack repository**
   ```sh
   git clone https://github.com/ifzhang/ByteTrack.git
   cd ByteTrack
   pip install -r requirements.txt
   python setup.py develop    # or `pip install -e .`
   pip install cython_bbox    # optional helper used by the tracker
   ```
   > Alternatively you can install directly from GitHub:
   > `pip install git+https://github.com/ifzhang/ByteTrack.git`

2. **Verify the import**
   ```python
   from yolox.tracker.byte_tracker import BYTETracker
   ```
   If this fails, make sure the `yolox` package from the ByteTrack repo is on
   your `PYTHONPATH` or that the repository was installed correctly.

3. **Restart the Flask application**
   ByteTrack is loaded during module import; restarting ensures the trackers
   are constructed with the correct arguments.

## How it works in `app.py`

* At startup the code tries to import `BYTETracker` and, if successful,
  instantiates two trackers (`vehicle_tracker` and `emergency_tracker`).
* During every frame the YOLO models still perform detection, but the
  `track` call is skipped.  Detected boxes are packaged as `[x1,y1,x2,y2,score]`
  arrays and passed to `BYTETracker.update`.  The returned `STrack` objects
  provide `track_id` and `tlbr` coordinates that are drawn on the output.
* If `BYTETracker` cannot be imported the application automatically falls
  back to the original `model.track(...)` behavior so nothing breaks.

## Notes & Tips

* ByteTrack is class-agnostic; for this app we run two separate trackers
  (one for normal vehicles and one for emergency vehicles) so that IDs
  do not collide.
* The tracking history used for traffic jam detection is only maintained
  for the non-emergency tracker.
* Make sure that the camera resolution and inference speed are sufficient
  to maintain real‑time performance; ByteTrack adds overhead but is still
  very fast on modern CPUs/GPUs.

---

If you run into installation problems, consult the official
[ByteTrack README](https://github.com/ifzhang/ByteTrack) or raise an issue
on the repository.