import os
import re
import time
import math
import threading
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ---------------------------------------------------------------------------
# Scalable analytics settings
# ---------------------------------------------------------------------------
MAX_CAMERAS = max(4, min(32, int(os.getenv("MAX_CAMERAS", "32"))))
ANALYTICS_FPS = max(0.5, float(os.getenv("ANALYTICS_FPS", "3")))
ANALYTICS_INTERVAL = 1.0 / ANALYTICS_FPS
ANALYTICS_BATCH_SIZE = max(1, int(os.getenv("ANALYTICS_BATCH_SIZE", "8")))
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.22"))
DETECTION_IMGSZ = max(416, int(os.getenv("DETECTION_IMGSZ", "640")))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.55"))
MAX_DETECTIONS = max(20, int(os.getenv("MAX_DETECTIONS", "120")))
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto").strip()
PASS_MOVEMENT_RATIO = float(os.getenv("PASS_MOVEMENT_RATIO", "0.035"))
PASS_MIN_TRACK_SECONDS = float(os.getenv("PASS_MIN_TRACK_SECONDS", "0.35"))
TRACK_TTL_SECONDS = float(os.getenv("TRACK_TTL_SECONDS", "2.0"))
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.20"))
TRACK_CENTER_RATIO = float(os.getenv("TRACK_CENTER_RATIO", "0.12"))
STOPPED_SECONDS = int(os.getenv("STOPPED_SECONDS", "240"))
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").lower() == "true"
PLATE_OCR_INTERVAL = max(1.0, float(os.getenv("PLATE_OCR_INTERVAL", "2.0")))
JPEG_QUALITY = min(92, max(50, int(os.getenv("JPEG_QUALITY", "76"))))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26s.pt").strip() or "yolo26s.pt"
DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent / "cctv.db"))

ALLOWED_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def load_camera_config():
    cameras = []
    for camera_id in range(1, MAX_CAMERAS + 1):
        legacy = os.getenv("CAMERA_RTSP_URL", "") if camera_id == 1 else ""
        url = os.getenv(f"CAMERA{camera_id}_RTSP_URL", legacy).strip()
        name = os.getenv(f"CAMERA{camera_id}_NAME", f"Camera {camera_id}").strip() or f"Camera {camera_id}"
        if camera_id <= 4 or url:
            cameras.append({"id": camera_id, "name": name, "url": url})
    return cameras


CAMERAS = load_camera_config()
app = FastAPI(title="AutoGearUp CCTV Gateway")


# ---------------------------------------------------------------------------
# Database: metadata/events only. No video is stored by AutoGearUp.
# ---------------------------------------------------------------------------
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          camera INTEGER NOT NULL,
          kind TEXT NOT NULL,
          vehicle_type TEXT,
          track_id INTEGER,
          direction TEXT,
          confidence REAL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);

        CREATE TABLE IF NOT EXISTS plates (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          camera INTEGER NOT NULL,
          plate TEXT NOT NULL,
          confidence REAL,
          vehicle_type TEXT,
          track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_plates_ts ON plates(ts);
        CREATE INDEX IF NOT EXISTS idx_plates_camera ON plates(camera);

        CREATE TABLE IF NOT EXISTS incidents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          camera INTEGER NOT NULL,
          type TEXT NOT NULL,
          severity TEXT,
          description TEXT,
          track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts);
        CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera);
        """)


def save_event(event):
    with db() as con:
        con.execute(
            "INSERT INTO events(ts,camera,kind,vehicle_type,track_id,direction,confidence) VALUES(?,?,?,?,?,?,?)",
            (
                event["time"],
                event["camera"],
                "person" if event["type"] == "person" else "vehicle",
                event["type"],
                event.get("track_id"),
                event.get("direction"),
                event.get("confidence"),
            ),
        )


def save_plate(plate):
    with db() as con:
        con.execute(
            "INSERT INTO plates(ts,camera,plate,confidence,vehicle_type,track_id) VALUES(?,?,?,?,?,?)",
            (
                plate["time"],
                plate["camera"],
                plate["plate"],
                plate.get("confidence"),
                plate.get("vehicle_type"),
                plate.get("track_id"),
            ),
        )


def save_incident(incident):
    with db() as con:
        con.execute(
            "INSERT INTO incidents(ts,camera,type,severity,description,track_id) VALUES(?,?,?,?,?,?)",
            (
                incident["time"],
                incident["camera"],
                incident["type"],
                incident.get("severity"),
                incident.get("description"),
                incident.get("track_id"),
            ),
        )


# ---------------------------------------------------------------------------
# Lightweight per-camera tracker.
# The detector is shared/batched; only tracking state is per camera.
# ---------------------------------------------------------------------------
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


class LightweightTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_shape):
        now = time.time()
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))

        # Remove stale tracks before matching.
        stale = [
            tid for tid, track in self.tracks.items()
            if now - track["last_seen"] > TRACK_TTL_SECONDS
        ]
        for tid in stale:
            self.tracks.pop(tid, None)

        unmatched_tracks = set(self.tracks.keys())
        enriched = []

        # Match high-confidence detections first.
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        for det in detections:
            box = det["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            best_tid = None
            best_score = -1.0

            for tid in list(unmatched_tracks):
                track = self.tracks[tid]
                if track["type"] != det["type"]:
                    continue

                iou = bbox_iou(box, track["box"])
                tx, ty = track["center"]
                center_distance = math.hypot(cx - tx, cy - ty) / diag

                # IoU is preferred; center proximity helps fast-moving objects.
                if iou >= TRACK_IOU_THRESHOLD:
                    score = 1.0 + iou
                elif center_distance <= TRACK_CENTER_RATIO:
                    score = 1.0 - center_distance
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1
                self.tracks[best_tid] = {
                    "id": best_tid,
                    "type": det["type"],
                    "box": box,
                    "center": (cx, cy),
                    "first_center": (cx, cy),
                    "first_seen": now,
                    "last_seen": now,
                    "last_center": (cx, cy),
                    "counted": False,
                    "still_since": now,
                    "incident_latched": False,
                    "last_plate_ocr": 0.0,
                }
            else:
                unmatched_tracks.discard(best_tid)
                track = self.tracks[best_tid]
                track["box"] = box
                track["last_seen"] = now
                track["center"] = (cx, cy)

            det = dict(det)
            det["track_id"] = best_tid
            enriched.append(det)

        return enriched, self.tracks


# ---------------------------------------------------------------------------
# Camera capture workers: capture only. They do not load AI models.
# ---------------------------------------------------------------------------
class CameraWorker:
    def __init__(self, camera_id, name, rtsp_url):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.frame = None
        self.frame_lock = threading.Lock()
        self.frame_version = 0
        self.running = False
        self.connected = False
        self.status = "not configured" if not rtsp_url else "starting"

        self.tracker = LightweightTracker()
        self.last_detections = []
        self.current_people = 0
        self.current_vehicles = 0
        self.last_inference_at = None
        self.analytics_running = False
        self.model_name = None

        self.people_passed = 0
        self.vehicles_passed = 0
        self.people_seen = 0
        self.vehicles_seen = 0
        self.seen_track_ids = set()
        self.counted_track_ids = set()
        self.vehicle_types = defaultdict(int)

        self.events = deque(maxlen=250)
        self.plates = deque(maxlen=150)
        self.incidents = deque(maxlen=150)
        self.readable_plates = 0

    def start(self):
        if self.running or not self.rtsp_url:
            return
        self.running = True
        threading.Thread(
            target=self.capture_loop,
            daemon=True,
            name=f"camera-capture-{self.camera_id}",
        ).start()

    def open_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def capture_loop(self):
        while self.running:
            self.status = "connecting"
            cap = self.open_capture()

            if not cap.isOpened():
                self.connected = False
                self.status = "retrying"
                print(f"[Camera {self.camera_id}] unable to open RTSP; retrying")
                time.sleep(3)
                continue

            self.connected = True
            self.status = "online"
            print(f"[Camera {self.camera_id}] stream connected")

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.connected = False
                    self.status = "reconnecting"
                    break

                with self.frame_lock:
                    self.frame = frame
                    self.frame_version += 1

            cap.release()
            time.sleep(1)

    def snapshot(self):
        with self.frame_lock:
            if self.frame is None:
                return None, self.frame_version
            return self.frame.copy(), self.frame_version

    def consume_detections(self, detections, frame):
        now = time.time()
        h, w = frame.shape[:2]
        diag = max(1.0, math.hypot(w, h))
        tracked, tracks = self.tracker.update(detections, frame.shape)

        self.current_people = sum(1 for d in tracked if d["type"] == "person")
        self.current_vehicles = sum(1 for d in tracked if d["type"] in VEHICLE_CLASSES)
        self.last_detections = tracked
        self.last_inference_at = time.strftime("%Y-%m-%d %H:%M:%S")

        for det in tracked:
            tid = det["track_id"]
            object_type = det["type"]
            track = tracks[tid]

            if tid not in self.seen_track_ids:
                self.seen_track_ids.add(tid)
                if object_type == "person":
                    self.people_seen += 1
                elif object_type in VEHICLE_CLASSES:
                    self.vehicles_seen += 1

            fx, fy = track["first_center"]
            cx, cy = track["center"]
            movement = math.hypot(cx - fx, cy - fy)
            movement_threshold = max(16.0, diag * PASS_MOVEMENT_RATIO)
            age = now - track["first_seen"]

            if (
                tid not in self.counted_track_ids
                and age >= PASS_MIN_TRACK_SECONDS
                and movement >= movement_threshold
            ):
                self.counted_track_ids.add(tid)
                track["counted"] = True

                if object_type == "person":
                    self.people_passed += 1
                else:
                    self.vehicles_passed += 1
                    self.vehicle_types[object_type] += 1

                if abs(cy - fy) >= abs(cx - fx):
                    direction = "inbound" if cy > fy else "outbound"
                else:
                    direction = "right" if cx > fx else "left"

                event = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "camera": self.camera_id,
                    "type": object_type,
                    "track_id": tid,
                    "direction": direction,
                    "confidence": round(float(det["confidence"]), 3),
                }
                self.events.appendleft(event)
                save_event(event)

            # Stopped vehicle incident logic.
            if object_type in VEHICLE_CLASSES:
                lx, ly = track["last_center"]
                local_move = math.hypot(cx - lx, cy - ly)

                if local_move < max(4.0, diag * 0.002):
                    if now - track["still_since"] >= STOPPED_SECONDS and not track["incident_latched"]:
                        track["incident_latched"] = True
                        incident = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": self.camera_id,
                            "type": "stopped_vehicle",
                            "severity": "medium",
                            "description": f"{object_type.title()} stationary for {STOPPED_SECONDS}s",
                            "track_id": tid,
                        }
                        self.incidents.appendleft(incident)
                        save_incident(incident)
                else:
                    track["still_since"] = now

                track["last_center"] = (cx, cy)

        return tracked

    def draw_overlay(self, frame):
        h, _ = frame.shape[:2]
        for det in self.last_detections:
            x1, y1, x2, y2 = map(int, det["box"])
            is_vehicle = det["type"] in VEHICLE_CLASSES
            colour = (255, 170, 80) if is_vehicle else (80, 220, 170)
            label = f'{det["type"]} {det["confidence"]:.0%} ID:{det["track_id"]}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .45,
                colour,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Now: {self.current_people} people | {self.current_vehicles} vehicles",
            (12, max(42, h - 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .55,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Vehicles passed: {self.vehicles_passed} / {self.vehicles_seen} seen",
            (12, max(24, h - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .55,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return frame


# ---------------------------------------------------------------------------
# Shared batched AI engine.
# One model services every camera; frames are grouped into batches.
# ---------------------------------------------------------------------------
class BatchAnalyticsEngine:
    def __init__(self, workers):
        self.workers = workers
        self.model = None
        self.model_name = YOLO_MODEL
        self.running = False
        self.status = "starting"
        self.last_error = None
        self.last_batch_ms = None
        self.last_batch_size = 0
        self.total_batches = 0
        self.total_frames = 0
        self.last_versions = defaultdict(int)
        self.ocr = None
        self.ocr_lock = threading.Lock()

    def choose_device(self):
        if INFERENCE_DEVICE != "auto":
            return INFERENCE_DEVICE
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    def load_model(self):
        try:
            from ultralytics import YOLO
            candidates = []
            for candidate in (
                YOLO_MODEL,
                "yolo26s.pt",
                "yolo26n.pt",
                "yolo11s.pt",
                "yolo11n.pt",
            ):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)

            for candidate in candidates:
                try:
                    self.model = YOLO(candidate)
                    self.model_name = candidate
                    self.status = "ready"
                    print(
                        f"[Analytics] shared model loaded: {candidate}; "
                        f"device={self.choose_device()} batch={ANALYTICS_BATCH_SIZE} "
                        f"imgsz={DETECTION_IMGSZ} target_fps/camera={ANALYTICS_FPS}"
                    )
                    break
                except Exception as e:
                    print(f"[Analytics] could not load {candidate}: {e}")

            if self.model is None:
                self.status = "model unavailable"
                return

            if ENABLE_PLATE_OCR:
                try:
                    import easyocr
                    self.ocr = easyocr.Reader(["en"], gpu=False)
                    print("[Analytics] shared EasyOCR reader loaded")
                except Exception as e:
                    print(f"[Analytics] plate OCR disabled: {e}")
        except Exception as e:
            self.status = "ultralytics unavailable"
            self.last_error = str(e)
            print(f"[Analytics] unavailable: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(
            target=self.run,
            daemon=True,
            name="shared-batch-analytics",
        ).start()

    def maybe_plate_ocr(self, worker, frame, tracked):
        if self.ocr is None:
            return

        now = time.time()
        for det in tracked:
            if det["type"] not in VEHICLE_CLASSES:
                continue

            tid = det.get("track_id")
            if tid is None:
                continue

            track = worker.tracker.tracks.get(tid)
            if not track:
                continue

            if now - track.get("last_plate_ocr", 0.0) < PLATE_OCR_INTERVAL:
                continue
            track["last_plate_ocr"] = now

            x1, y1, x2, y2 = map(int, det["box"])
            vh = max(1, y2 - y1)
            # Most plates are in the lower half of the vehicle bounding box.
            crop = frame[y1 + vh // 2:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                with self.ocr_lock:
                    reads = self.ocr.readtext(crop, detail=1, paragraph=False)
            except Exception:
                continue

            for _, text_value, score in reads:
                clean = re.sub(r"[^A-Z0-9]", "", str(text_value).upper())
                if not (4 <= len(clean) <= 8 and float(score) >= 0.50):
                    continue

                duplicate = any(
                    p.get("plate") == clean and p.get("track_id") == tid
                    for p in list(worker.plates)[:20]
                )
                if duplicate:
                    break

                plate_event = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "camera": worker.camera_id,
                    "plate": clean,
                    "confidence": round(float(score), 3),
                    "vehicle_type": det["type"],
                    "track_id": tid,
                }
                worker.readable_plates += 1
                worker.plates.appendleft(plate_event)
                save_plate(plate_event)
                break

    def run(self):
        self.load_model()
        if self.model is None:
            return

        while self.running:
            cycle_start = time.time()
            samples = []

            for worker in self.workers.values():
                if not worker.rtsp_url or not worker.connected:
                    continue
                frame, version = worker.snapshot()
                if frame is None:
                    continue
                if version == self.last_versions[worker.camera_id]:
                    continue
                self.last_versions[worker.camera_id] = version
                samples.append((worker, frame))

            # Keep latency low: process the freshest frame from each camera,
            # chunked to the configured batch size.
            for offset in range(0, len(samples), ANALYTICS_BATCH_SIZE):
                chunk = samples[offset:offset + ANALYTICS_BATCH_SIZE]
                if not chunk:
                    continue

                workers_chunk = [x[0] for x in chunk]
                frames = [x[1] for x in chunk]

                started = time.perf_counter()
                try:
                    results = self.model.predict(
                        source=frames,
                        verbose=False,
                        classes=list(ALLOWED_CLASSES.keys()),
                        conf=DETECTION_CONFIDENCE,
                        iou=DETECTION_IOU,
                        imgsz=DETECTION_IMGSZ,
                        max_det=MAX_DETECTIONS,
                        device=self.choose_device(),
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    self.last_batch_ms = round(elapsed_ms, 1)
                    self.last_batch_size = len(frames)
                    self.total_batches += 1
                    self.total_frames += len(frames)
                    self.status = "running"

                    for worker, frame, result in zip(workers_chunk, frames, results):
                        detections = []
                        boxes = result.boxes

                        if boxes is not None:
                            xyxy = boxes.xyxy.cpu().numpy()
                            classes = boxes.cls.int().cpu().tolist()
                            confs = boxes.conf.cpu().numpy().tolist()

                            for bb, cls_id, confidence in zip(xyxy, classes, confs):
                                if cls_id not in ALLOWED_CLASSES:
                                    continue
                                detections.append({
                                    "box": tuple(map(float, bb)),
                                    "type": ALLOWED_CLASSES[cls_id],
                                    "confidence": float(confidence),
                                })

                        worker.analytics_running = True
                        worker.model_name = self.model_name
                        tracked = worker.consume_detections(detections, frame)
                        self.maybe_plate_ocr(worker, frame, tracked)

                except Exception as e:
                    self.last_error = str(e)
                    self.status = "error"
                    print(f"[Analytics] batch error: {e}")

            elapsed = time.time() - cycle_start
            time.sleep(max(0.01, ANALYTICS_INTERVAL - elapsed))

    def health(self):
        effective_fps = 0.0
        if self.last_batch_ms and self.last_batch_ms > 0:
            effective_fps = round((1000.0 / self.last_batch_ms) * max(1, self.last_batch_size), 1)
        return {
            "mode": "shared_batch",
            "status": self.status,
            "model": self.model_name if self.model is not None else None,
            "device": self.choose_device(),
            "batch_size": ANALYTICS_BATCH_SIZE,
            "target_fps_per_camera": ANALYTICS_FPS,
            "imgsz": DETECTION_IMGSZ,
            "confidence": DETECTION_CONFIDENCE,
            "last_batch_ms": self.last_batch_ms,
            "last_batch_size": self.last_batch_size,
            "estimated_inference_fps": effective_fps,
            "total_batches": self.total_batches,
            "total_frames": self.total_frames,
            "last_error": self.last_error,
        }


workers = {
    camera["id"]: CameraWorker(camera["id"], camera["name"], camera["url"])
    for camera in CAMERAS
}
analytics = BatchAnalyticsEngine(workers)


@app.on_event("startup")
def startup():
    init_db()
    for worker in workers.values():
        worker.start()
    analytics.start()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    configured = [w for w in workers.values() if w.rtsp_url]
    online = [w for w in configured if w.connected]
    return {
        "ok": True,
        "camera_configured": bool(configured),
        "camera_count": len(configured),
        "online_count": len(online),
        "analytics_enabled": analytics.model is not None,
        "plate_ocr_enabled": analytics.ocr is not None,
        "analytics": analytics.health(),
        "cameras": [
            {
                "id": w.camera_id,
                "name": w.name,
                "configured": bool(w.rtsp_url),
                "online": w.connected,
                "status": w.status,
                "frame_available": w.frame is not None,
                "analytics_enabled": analytics.model is not None,
                "analytics_running": w.analytics_running,
                "model": w.model_name,
                "detection_imgsz": DETECTION_IMGSZ,
                "detection_confidence": DETECTION_CONFIDENCE,
                "recognition_profile": "shared-batch",
                "current_people": w.current_people,
                "current_vehicles": w.current_vehicles,
                "vehicles_passed": w.vehicles_passed,
                "vehicles_seen": w.vehicles_seen,
                "people_passed": w.people_passed,
                "people_seen": w.people_seen,
                "last_inference_at": w.last_inference_at,
            }
            for w in workers.values()
        ],
    }


@app.get("/api/cameras")
def cameras():
    return health()["cameras"]


def period_bounds(period, start=None, end=None):
    now = datetime.now()

    if period == "today":
        a = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        a = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif period == "month":
        a = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "year":
        a = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "custom" and start:
        a = datetime.fromisoformat(start).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        a = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "custom" and end:
        b = datetime.fromisoformat(end).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    else:
        b = now

    return (
        a.strftime("%Y-%m-%d %H:%M:%S"),
        b.strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.get("/api/history")
def history(period: str = "today", start: str | None = None, end: str | None = None):
    a, b = period_bounds(period, start, end)
    names = {camera["id"]: camera["name"] for camera in CAMERAS}

    with db() as con:
        people = con.execute(
            "SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='person'",
            (a, b),
        ).fetchone()["n"]
        vehicles = con.execute(
            "SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='vehicle'",
            (a, b),
        ).fetchone()["n"]
        plates_n = con.execute(
            "SELECT COUNT(*) n FROM plates WHERE ts BETWEEN ? AND ?",
            (a, b),
        ).fetchone()["n"]
        incidents_n = con.execute(
            "SELECT COUNT(*) n FROM incidents WHERE ts BETWEEN ? AND ?",
            (a, b),
        ).fetchone()["n"]

        vehicle_types = {
            row["vehicle_type"]: row["n"]
            for row in con.execute(
                "SELECT vehicle_type, COUNT(*) n FROM events "
                "WHERE ts BETWEEN ? AND ? AND kind='vehicle' GROUP BY vehicle_type",
                (a, b),
            )
        }

        latest_events = [
            dict(row)
            for row in con.execute(
                "SELECT ts as time,camera,vehicle_type as type,track_id,direction,confidence "
                "FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100",
                (a, b),
            )
        ]
        latest_plates = [
            dict(row)
            for row in con.execute(
                "SELECT ts as time,camera,plate,confidence,vehicle_type,track_id "
                "FROM plates WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100",
                (a, b),
            )
        ]
        latest_incidents = [
            dict(row)
            for row in con.execute(
                "SELECT ts as time,camera,type,severity,description,track_id "
                "FROM incidents WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100",
                (a, b),
            )
        ]

    for row in latest_events:
        row["camera_name"] = names.get(row["camera"], f"Camera {row['camera']}")
    for row in latest_plates:
        row["camera_name"] = names.get(row["camera"], f"Camera {row['camera']}")
    for row in latest_incidents:
        row["camera_name"] = names.get(row["camera"], f"Camera {row['camera']}")

    return {
        "period": period,
        "start": a,
        "end": b,
        "people_passed": people,
        "vehicles_passed": vehicles,
        "readable_plates": plates_n,
        "major_incidents": incidents_n,
        "vehicle_types": vehicle_types,
        "latest_events": latest_events,
        "latest_plates": latest_plates,
        "incidents": latest_incidents,
    }


@app.get("/api/stats")
def stats():
    return history("today")


def mjpeg(worker):
    while True:
        frame, _ = worker.snapshot()

        if frame is None:
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"{worker.name}: {worker.status}",
                (60, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                .8,
                (220, 220, 220),
                2,
            )
        else:
            frame = worker.draw_overlay(frame)

        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg.tobytes()
                + b"\r\n"
            )

        # Streaming is on-demand only. No recording is performed here.
        time.sleep(0.05)


@app.get("/api/cameras/{camera_id}/stream")
def stream(camera_id: int):
    worker = workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not worker.rtsp_url:
        raise HTTPException(status_code=404, detail="Camera is not configured")

    return StreamingResponse(
        mjpeg(worker),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


BASE = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="dashboard")
