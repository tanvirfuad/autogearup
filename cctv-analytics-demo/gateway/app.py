import os, re, time, threading, sqlite3
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

DETECT_EVERY = max(1, int(os.getenv("DETECT_EVERY", "2")))
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.30"))
COUNT_LINE_Y = float(os.getenv("COUNT_LINE_Y", "0.60"))
COUNT_HYSTERESIS = float(os.getenv("COUNT_HYSTERESIS", "0.06"))
PASS_MOVEMENT_RATIO = float(os.getenv("PASS_MOVEMENT_RATIO", "0.035"))
PASS_MIN_TRACK_SECONDS = float(os.getenv("PASS_MIN_TRACK_SECONDS", "0.35"))
STOPPED_SECONDS = int(os.getenv("STOPPED_SECONDS", "240"))
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").lower() == "true"
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent / "cctv.db"))

CAMERAS = [
    {
        "id": 1,
        "name": os.getenv("CAMERA1_NAME", "Camera 1"),
        "url": os.getenv("CAMERA1_RTSP_URL", os.getenv("CAMERA_RTSP_URL", "")),
    },
    {
        "id": 2,
        "name": os.getenv("CAMERA2_NAME", "Camera 2"),
        "url": os.getenv("CAMERA2_RTSP_URL", ""),
    },
    {
        "id": 3,
        "name": os.getenv("CAMERA3_NAME", "Camera 3"),
        "url": os.getenv("CAMERA3_RTSP_URL", ""),
    },
    {
        "id": 4,
        "name": os.getenv("CAMERA4_NAME", "Camera 4"),
        "url": os.getenv("CAMERA4_RTSP_URL", ""),
    },
]

app = FastAPI(title="AutoGearUp CCTV Gateway")


def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, camera INTEGER NOT NULL,
          kind TEXT NOT NULL, vehicle_type TEXT, track_id INTEGER, direction TEXT, confidence REAL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);

        CREATE TABLE IF NOT EXISTS plates (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, camera INTEGER NOT NULL,
          plate TEXT NOT NULL, confidence REAL, vehicle_type TEXT, track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_plates_ts ON plates(ts);
        CREATE INDEX IF NOT EXISTS idx_plates_camera ON plates(camera);

        CREATE TABLE IF NOT EXISTS incidents (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, camera INTEGER NOT NULL,
          type TEXT NOT NULL, severity TEXT, description TEXT, track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts);
        CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera);
        """)


def save_event(e):
    with db() as con:
        con.execute(
            "INSERT INTO events(ts,camera,kind,vehicle_type,track_id,direction,confidence) VALUES(?,?,?,?,?,?,?)",
            (
                e["time"], e["camera"],
                "person" if e["type"] == "person" else "vehicle",
                e["type"], e.get("track_id"), e.get("direction"), e.get("confidence")
            )
        )


def save_plate(p):
    with db() as con:
        con.execute(
            "INSERT INTO plates(ts,camera,plate,confidence,vehicle_type,track_id) VALUES(?,?,?,?,?,?)",
            (
                p["time"], p["camera"], p["plate"], p.get("confidence"),
                p.get("vehicle_type"), p.get("track_id")
            )
        )


def save_incident(i):
    with db() as con:
        con.execute(
            "INSERT INTO incidents(ts,camera,type,severity,description,track_id) VALUES(?,?,?,?,?,?)",
            (
                i["time"], i["camera"], i["type"], i.get("severity"),
                i.get("description"), i.get("track_id")
            )
        )


class CameraWorker:
    def __init__(self, camera_id, name, rtsp_url):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.connected = False
        self.status = "not configured" if not rtsp_url else "starting"
        self.model = None
        self.ocr = None
        self.frame_no = 0
        self.prev_side = {}
        self.first_pos = {}
        self.first_seen = {}
        self.counted_tracks = set()
        self.last_seen = {}
        self.last_detections = []
        self.current_people = 0
        self.current_vehicles = 0
        self.last_inference_at = None
        self.last_pos = {}
        self.still_since = {}
        self.incident_latched = set()
        self.people_passed = 0
        self.vehicles_passed = 0
        self.readable_plates = 0
        self.vehicle_types = defaultdict(int)
        self.plates = deque(maxlen=100)
        self.incidents = deque(maxlen=100)
        self.events = deque(maxlen=200)

    def load_models(self):
        if not self.rtsp_url:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            print(f"[Camera {self.camera_id}] YOLO loaded")
        except Exception as e:
            print(f"[Camera {self.camera_id}] YOLO unavailable; live video will still work: {e}")

        if ENABLE_PLATE_OCR:
            try:
                import easyocr
                self.ocr = easyocr.Reader(["en"], gpu=False)
                print(f"[Camera {self.camera_id}] EasyOCR loaded")
            except Exception as e:
                print(f"[Camera {self.camera_id}] Plate OCR disabled: {e}")

    def start(self):
        if self.running or not self.rtsp_url:
            return
        self.running = True
        threading.Thread(target=self.run, daemon=True, name=f"camera-{self.camera_id}").start()

    def open_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def run(self):
        self.load_models()
        while self.running:
            self.status = "connecting"
            cap = self.open_capture()

            if not cap.isOpened():
                self.connected = False
                self.status = "retrying"
                print(f"[Camera {self.camera_id}] Unable to open RTSP stream; retrying...")
                time.sleep(3)
                continue

            self.connected = True
            self.status = "online"
            print(f"[Camera {self.camera_id}] Stream connected")

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.connected = False
                    self.status = "reconnecting"
                    break

                self.frame_no += 1
                if self.model:
                    if self.frame_no % DETECT_EVERY == 0:
                        frame = self.analyze(frame)
                    else:
                        frame = self.draw_last_detections(frame)

                with self.frame_lock:
                    self.frame = frame

            cap.release()
            time.sleep(1)

    def draw_last_detections(self, frame):
        h, w = frame.shape[:2]
        line_y = int(h * COUNT_LINE_Y)
        band = max(8, int(h * COUNT_HYSTERESIS))
        upper = max(0, line_y - band)
        lower = min(h - 1, line_y + band)

        cv2.line(frame, (0, upper), (w, upper), (60, 120, 180), 1)
        cv2.line(frame, (0, line_y), (w, line_y), (80, 220, 170), 2)
        cv2.line(frame, (0, lower), (w, lower), (60, 120, 180), 1)

        for d in self.last_detections:
            x1, y1, x2, y2 = d["box"]
            colour = d["colour"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                d["label"],
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
            (12, max(24, h - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .55,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return frame

    def analyze(self, frame):
        h, w = frame.shape[:2]
        line_y = int(h * COUNT_LINE_Y)
        band = max(8, int(h * COUNT_HYSTERESIS))
        upper = max(0, line_y - band)
        lower = min(h - 1, line_y + band)

        allowed = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        self.current_people = 0
        self.current_vehicles = 0
        fresh_detections = []

        try:
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                classes=list(allowed.keys()),
                tracker="bytetrack.yaml",
                conf=DETECTION_CONFIDENCE,
                iou=0.50,
            )

            if not results:
                self.last_detections = []
                return self.draw_last_detections(frame)

            boxes = results[0].boxes
            if boxes is None:
                self.last_detections = []
                return self.draw_last_detections(frame)

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(classes)

            for bb, cls_id, confidence, track_id in zip(xyxy, classes, confs, ids):
                if cls_id not in allowed:
                    continue

                x1, y1, x2, y2 = map(int, bb)
                object_type = allowed[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                is_vehicle = object_type != "person"

                if is_vehicle:
                    self.current_vehicles += 1
                else:
                    self.current_people += 1

                colour = (255, 170, 80) if is_vehicle else (80, 220, 170)
                label = f"{object_type} {confidence:.0%}"
                if track_id is not None:
                    label += f" ID:{track_id}"

                fresh_detections.append({
                    "box": (x1, y1, x2, y2),
                    "colour": colour,
                    "label": label,
                })

                if track_id is None:
                    continue

                # "Passed" now means a unique tracked person/vehicle moved
                # meaningfully within this camera. No line crossing is required.
                # Each track contributes at most one count per camera.
                now = time.time()
                self.last_seen[track_id] = now
                self.first_pos.setdefault(track_id, (cx, cy))
                self.first_seen.setdefault(track_id, now)

                fx, fy = self.first_pos[track_id]
                movement = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
                movement_threshold = max(18.0, ((w * w + h * h) ** 0.5) * PASS_MOVEMENT_RATIO)
                age = now - self.first_seen[track_id]

                if (
                    track_id not in self.counted_tracks
                    and age >= PASS_MIN_TRACK_SECONDS
                    and movement >= movement_threshold
                ):
                    self.counted_tracks.add(track_id)

                    if object_type == "person":
                        self.people_passed += 1
                    else:
                        self.vehicles_passed += 1
                        self.vehicle_types[object_type] += 1

                    direction = "moving"
                    if abs(cy - fy) >= abs(cx - fx):
                        direction = "inbound" if cy > fy else "outbound"
                    else:
                        direction = "right" if cx > fx else "left"

                    event = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera": self.camera_id,
                        "type": object_type,
                        "track_id": track_id,
                        "direction": direction,
                        "confidence": round(float(confidence), 3),
                    }
                    self.events.appendleft(event)
                    save_event(event)

                # Keep the previous side only for the visual counting guide.
                if cy < upper:
                    self.prev_side[track_id] = -1
                elif cy > lower:
                    self.prev_side[track_id] = 1

                if is_vehicle:
                    previous_position = self.last_pos.get(track_id)
                    now = time.time()

                    if previous_position:
                        distance = (
                            (cx - previous_position[0]) ** 2
                            + (cy - previous_position[1]) ** 2
                        ) ** 0.5

                        if distance < 10:
                            self.still_since.setdefault(track_id, now)

                            if (
                                now - self.still_since[track_id] >= STOPPED_SECONDS
                                and track_id not in self.incident_latched
                            ):
                                self.incident_latched.add(track_id)
                                incident = {
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "camera": self.camera_id,
                                    "type": "stopped_vehicle",
                                    "severity": "medium",
                                    "description": f"{object_type.title()} stationary for {STOPPED_SECONDS}s",
                                    "track_id": track_id,
                                }
                                self.incidents.appendleft(incident)
                                save_incident(incident)
                        else:
                            self.still_since.pop(track_id, None)

                    self.last_pos[track_id] = (cx, cy)

                    if (
                        self.ocr
                        and confidence > 0.65
                        and (self.frame_no // DETECT_EVERY) % 8 == 0
                    ):
                        self.try_plate_ocr(
                            frame, x1, y1, x2, y2, object_type, track_id
                        )

            # Drop stale tracker bookkeeping after 2 minutes.
            now = time.time()
            stale = [tid for tid, ts in self.last_seen.items() if now - ts > 120]
            for tid in stale:
                self.last_seen.pop(tid, None)
                self.first_pos.pop(tid, None)
                self.first_seen.pop(tid, None)
                self.prev_side.pop(tid, None)
                self.last_pos.pop(tid, None)
                self.still_since.pop(tid, None)
                self.incident_latched.discard(tid)
                self.counted_tracks.discard(tid)

            self.last_detections = fresh_detections
            self.last_inference_at = time.strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            cv2.putText(
                frame,
                f"Analytics error: {type(e).__name__}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                .6,
                (0, 0, 255),
                2,
            )

        return self.draw_last_detections(frame)

    def try_plate_ocr(self, frame, x1, y1, x2, y2, vehicle_type, track_id):
        vh = max(1, y2 - y1)
        crop = frame[y1 + vh // 2:y2, x1:x2]

        if crop.size == 0:
            return

        try:
            detections = self.ocr.readtext(crop, detail=1, paragraph=False)

            for _, text, score in detections:
                clean = re.sub(r"[^A-Z0-9]", "", text.upper())

                if 4 <= len(clean) <= 8 and score >= 0.50:
                    duplicate = any(
                        p.get("plate") == clean and p.get("track_id") == track_id
                        for p in list(self.plates)[:10]
                    )

                    if not duplicate:
                        self.readable_plates += 1
                        plate_event = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": self.camera_id,
                            "plate": clean,
                            "confidence": round(float(score), 3),
                            "vehicle_type": vehicle_type,
                            "track_id": track_id,
                        }
                        self.plates.appendleft(plate_event)
                        save_plate(plate_event)

                    break

        except Exception:
            pass


workers = {
    camera["id"]: CameraWorker(camera["id"], camera["name"], camera["url"])
    for camera in CAMERAS
}


@app.on_event("startup")
def startup():
    init_db()
    for worker in workers.values():
        worker.start()


@app.get("/api/health")
def health():
    configured = [w for w in workers.values() if w.rtsp_url]
    return {
        "ok": True,
        "camera_configured": bool(configured),
        "camera_count": len(configured),
        "online_count": sum(1 for w in configured if w.connected),
        "analytics_enabled": any(w.model is not None for w in configured),
        "plate_ocr_enabled": any(w.ocr is not None for w in configured),
        "cameras": [
            {
                "id": w.camera_id,
                "name": w.name,
                "configured": bool(w.rtsp_url),
                "online": w.connected,
                "status": w.status,
                "frame_available": w.frame is not None,
                "analytics_enabled": w.model is not None,
                "current_people": w.current_people,
                "current_vehicles": w.current_vehicles,
                "last_inference_at": w.last_inference_at,
            }
            for w in workers.values()
        ],
    }


@app.get("/api/cameras")
def cameras():
    return [
        {
            "id": w.camera_id,
            "name": w.name,
            "configured": bool(w.rtsp_url),
            "online": w.connected,
            "status": w.status,
            "frame_available": w.frame is not None,
        }
        for w in workers.values()
    ]


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
        with worker.frame_lock:
            frame = None if worker.frame is None else worker.frame.copy()

        if frame is None:
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            message = f"{worker.name}: {worker.status}"
            cv2.putText(
                placeholder,
                message,
                (60, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                .8,
                (220, 220, 220),
                2,
            )
            frame = placeholder

        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 78],
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg.tobytes()
                + b"\r\n"
            )

        time.sleep(0.04)


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
    )


BASE = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="dashboard")
