# AutoGearUp Edge Analytics — 16 to 30 Camera Architecture

The gateway no longer creates one YOLO model per camera.

## Runtime pipeline

```text
Lorex NVR / cameras
       |
       | RTSP substreams
       v
Independent capture threads (1 per camera)
       |
       | latest frame only
       v
Shared batch scheduler
       |
       | 4/8/16 frames at a time
       v
ONE YOLO / TensorRT detector
       |
       v
Lightweight per-camera tracking
       |
       +--> people passed
       +--> vehicles passed
       +--> vehicle class
       +--> optional plate OCR
       +--> incident rules
       |
       v
SQLite event metadata + AutoGearUp API
```

Lorex remains the video recorder. AutoGearUp does not archive camera footage.

## Why this scales better

The previous prototype loaded a separate YOLO model for every camera. At 30 cameras that would waste large amounts of RAM/VRAM.

The shared-batch architecture:
- loads model weights once
- performs batch inference across multiple cameras
- samples each camera at a configurable AI rate
- keeps RTSP capture independent from inference
- only retains the newest frame in memory
- stores events/analytics, not video
- streams video to the browser only when a viewer opens it

Ultralytics supports inference on lists of images as a batch, and its multi-stream mode similarly uses batch size based on stream count.

## Suggested starting settings

### 4 cameras / existing test PC
```text
ANALYTICS_FPS=3
ANALYTICS_BATCH_SIZE=4
DETECTION_IMGSZ=640
YOLO_MODEL=yolo26s.pt
```

### 16 cameras / NVIDIA edge system
```text
ANALYTICS_FPS=3
ANALYTICS_BATCH_SIZE=8
DETECTION_IMGSZ=640
INFERENCE_DEVICE=0
```

### 24-30 cameras
```text
ANALYTICS_FPS=3
ANALYTICS_BATCH_SIZE=8
DETECTION_IMGSZ=640
INFERENCE_DEVICE=0
```

Then benchmark `/api/health`. It now reports:
- batch size
- batch inference time
- estimated inference FPS
- total frames analyzed
- model/device
- per-camera last inference time

If the detector cannot keep up, reduce `ANALYTICS_FPS`, lower `DETECTION_IMGSZ`, use `yolo26n.pt`, or move to TensorRT.

## TensorRT

On the actual NVIDIA deployment machine:

```bash
python export_tensorrt.py
```

The exporter creates an FP16 TensorRT engine with a dynamic batch dimension. Then set:

```text
YOLO_MODEL=yolo26s.engine
INFERENCE_DEVICE=0
```

TensorRT engines should be built on the target NVIDIA machine/runtime.

## No video storage

The AutoGearUp edge gateway stores only:
- timestamp
- camera ID
- person/vehicle event
- vehicle class
- direction/movement
- confidence
- plate text if OCR is enabled
- incident metadata

Lorex remains responsible for full video retention.
