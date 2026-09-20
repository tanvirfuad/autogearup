import os, re, time, threading, sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

LEGACY_RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")
DETECT_EVERY = int(os.getenv("DETECT_EVERY", "3"))
COUNT_LINE_Y = float(os.getenv("COUNT_LINE_Y", "0.60"))
STOPPED_SECONDS = int(os.getenv("STOPPED_SECONDS", "240"))
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").lower() == "true"
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent / "cctv.db"))

app = FastAPI(title="AutoGearUp CCTV Gateway")


class CameraCreate(BaseModel):
    name: str
    rtsp_url: str


def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS settings (\n          key TEXT PRIMARY KEY,\n          value TEXT\n        );\n        CREATE TABLE IF NOT EXISTS cameras (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          rtsp_url TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL
        );
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
        migrated = con.execute("SELECT value FROM settings WHERE key='legacy_camera_migrated'").fetchone()
        count = con.execute("SELECT COUNT(*) n FROM cameras").fetchone()["n"]
        if not migrated:
            if count == 0 and LEGACY_RTSP_URL:
                con.execute(
                    "INSERT INTO cameras(name,rtsp_url,enabled,created_at) VALUES(?,?,1,?)",
                    ("Camera 1", LEGACY_RTSP_URL, time.strftime("%Y-%m-%d %H:%M:%S"))
                )
            con.execute(
                "INSERT OR REPLACE INTO settings(key,value) VALUES('legacy_camera_migrated','1')"
            )


def save_event(e):
    with db() as con:
        con.execute(
            "INSERT INTO events(ts,camera,kind,vehicle_type,track_id,direction,confidence) VALUES(?,?,?,?,?,?,?)",
            (e["time"], e["camera"], "person" if e["type"] == "person" else "vehicle",
             e["type"], e.get("track_id"), e.get("direction"), e.get("confidence"))
        )


def save_plate(p):
    with db() as con:
        con.execute(
            "INSERT INTO plates(ts,camera,plate,confidence,vehicle_type,track_id) VALUES(?,?,?,?,?,?)",
            (p["time"], p["camera"], p["plate"], p.get("confidence"),
             p.get("vehicle_type"), p.get("track_id"))
        )


def save_incident(i):
    with db() as con:
        con.execute(
            "INSERT INTO incidents(ts,camera,type,severity,description,track_id) VALUES(?,?,?,?,?,?)",
            (i["time"], i["camera"], i["type"], i.get("severity"),
             i.get("description"), i.get("track_id"))
        )


def masked_rtsp(url):
    try:
        p = urlsplit(url)
        host = p.hostname or "camera"
        port = f":{p.port}" if p.port else ""
        return f"rtsp://***:***@{host}{port}{p.path or ''}"
    except Exception:
        return "rtsp://***"


class CameraWorker:
    def __init__(self, camera_id, name, rtsp_url):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.connected = False
        self.status = "starting"
        self.model = None
        self.ocr = None
        self.frame_no = 0
        self.prev_side = {}
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
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            print(f"[Camera {self.camera_id}] YOLO loaded")
        except Exception as e:
            print(f"[Camera {self.camera_id}] YOLO unavailable; video will still work: {e}")

        if ENABLE_PLATE_OCR:
            try:
                import easyocr
                self.ocr = easyocr.Reader(["en"], gpu=False)
                print(f"[Camera {self.camera_id}] EasyOCR loaded")
            except Exception as e:
                print(f"[Camera {self.camera_id}] Plate OCR disabled: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.run, daemon=True, name=f"camera-{self.camera_id}").start()

    def stop(self):
        self.running = False
        self.connected = False
        self.status = "stopped"

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
                if self.model and self.frame_no % DETECT_EVERY == 0:
                    frame = self.analyze(frame)

                with self.frame_lock:
                    self.frame = frame

            cap.release()
            time.sleep(1)

    def analyze(self, frame):
        h, w = frame.shape[:2]
        line_y = int(h * COUNT_LINE_Y)
        cv2.line(frame, (0, line_y), (w, line_y), (80, 220, 170), 2)
        allowed = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        try:
            results = self.model.track(
                frame, persist=True, verbose=False,
                classes=list(allowed.keys()), tracker="bytetrack.yaml"
            )
            if not results:
                return frame

            boxes = results[0].boxes
            if boxes is None:
                return frame

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(classes)

            for bb, cls_id, confidence, track_id in zip(xyxy, classes, confs, ids):
                if cls_id not in allowed:
                    continue

                x1, y1, x2, y2 = map(int, bb)
                name = allowed[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                is_vehicle = name != "person"
                colour = (255, 170, 80) if is_vehicle else (80, 220, 170)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{name} {confidence:.0%}" + (f" ID:{track_id}" if track_id is not None else ""),
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA
                )

                if track_id is None:
                    continue

                side = 1 if cy >= line_y else -1
                previous_side = self.prev_side.get(track_id)

                if previous_side is not None and previous_side != side:
                    if name == "person":
                        self.people_passed += 1
                    else:
                        self.vehicles_passed += 1
                        self.vehicle_types[name] += 1

                    event = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera": self.camera_id,
                        "type": name,
                        "track_id": track_id,
                        "direction": "inbound" if side == 1 else "outbound",
                        "confidence": round(float(confidence), 3)
                    }
                    self.events.appendleft(event)
                    save_event(event)

                self.prev_side[track_id] = side

                if is_vehicle:
                    previous_position = self.last_pos.get(track_id)
                    now = time.time()

                    if previous_position:
                        distance = ((cx - previous_position[0]) ** 2 + (cy - previous_position[1]) ** 2) ** 0.5
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
                                    "description": f"{name.title()} stationary for {STOPPED_SECONDS}s",
                                    "track_id": track_id
                                }
                                self.incidents.appendleft(incident)
                                save_incident(incident)
                        else:
                            self.still_since.pop(track_id, None)

                    self.last_pos[track_id] = (cx, cy)

                    if self.ocr and confidence > 0.65 and (self.frame_no // DETECT_EVERY) % 8 == 0:
                        self.try_plate_ocr(frame, x1, y1, x2, y2, name, track_id)

        except Exception as e:
            cv2.putText(
                frame, f"Analytics error: {type(e).__name__}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2
            )

        return frame

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
                            "track_id": track_id
                        }
                        self.plates.appendleft(plate_event)
                        save_plate(plate_event)
                    break
        except Exception:
            pass


workers = {}
workers_lock = threading.Lock()


def start_camera(row):
    camera_id = int(row["id"])
    with workers_lock:
        if camera_id in workers:
            return workers[camera_id]
        worker = CameraWorker(camera_id, row["name"], row["rtsp_url"])
        workers[camera_id] = worker
        worker.start()
        return worker


def start_saved_cameras():
    with db() as con:
        rows = con.execute("SELECT * FROM cameras WHERE enabled=1 ORDER BY id").fetchall()
    for row in rows:
        start_camera(row)


def camera_name_map():
    with db() as con:
        return {int(r["id"]): r["name"] for r in con.execute("SELECT id,name FROM cameras")}


@app.on_event("startup")
def startup():
    init_db()
    start_saved_cameras()


@app.get("/api/health")
def health():
    with workers_lock:
        current = list(workers.values())
    return {
        "ok": True,
        "camera_configured": bool(current),
        "camera_count": len(current),
        "online_count": sum(1 for w in current if w.connected),
        "analytics_enabled": any(w.model is not None for w in current),
        "plate_ocr_enabled": any(w.ocr is not None for w in current)
    }


@app.get("/api/cameras")
def list_cameras():
    with db() as con:
        rows = con.execute("SELECT id,name,enabled,created_at,rtsp_url FROM cameras ORDER BY id").fetchall()

    output = []
    with workers_lock:
        for row in rows:
            worker = workers.get(int(row["id"]))
            output.append({
                "id": int(row["id"]),
                "name": row["name"],
                "enabled": bool(row["enabled"]),
                "created_at": row["created_at"],
                "connection": masked_rtsp(row["rtsp_url"]),
                "status": worker.status if worker else "stopped",
                "online": bool(worker and worker.connected),
                "frame_available": bool(worker and worker.frame is not None),
                "analytics_enabled": bool(worker and worker.model is not None)
            })
    return output


@app.post("/api/cameras")
def add_camera(payload: CameraCreate):
    name = payload.name.strip() or "Camera"
    rtsp_url = payload.rtsp_url.strip()

    if not rtsp_url.lower().startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="Camera link must begin with rtsp://")

    with db() as con:
        cursor = con.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled,created_at) VALUES(?,?,1,?)",
            (name, rtsp_url, time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        camera_id = cursor.lastrowid
        row = con.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()

    worker = start_camera(row)
    return {
        "ok": True,
        "camera": {
            "id": camera_id,
            "name": name,
            "connection": masked_rtsp(rtsp_url),
            "status": worker.status
        }
    }


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int):
    with workers_lock:
        worker = workers.pop(camera_id, None)
    if worker:
        worker.stop()

    with db() as con:
        row = con.execute("SELECT id FROM cameras WHERE id=?", (camera_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Camera not found")
        con.execute("DELETE FROM cameras WHERE id=?", (camera_id,))

    return {"ok": True}


def period_bounds(period, start=None, end=None):
    now = datetime.now()

    if period == "today":
        a = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        a = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        a = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "year":
        a = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "custom" and start:
        a = datetime.fromisoformat(start).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        a = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "custom" and end:
        b = datetime.fromisoformat(end).replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        b = now

    return a.strftime("%Y-%m-%d %H:%M:%S"), b.strftime("%Y-%m-%d %H:%M:%S")


@app.get("/api/history")
def history(period: str = "today", start: str | None = None, end: str | None = None):
    a, b = period_bounds(period, start, end)

    with db() as con:
        people = con.execute(
            "SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='person'", (a, b)
        ).fetchone()["n"]
        vehicles = con.execute(
            "SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='vehicle'", (a, b)
        ).fetchone()["n"]
        plates_n = con.execute(
            "SELECT COUNT(*) n FROM plates WHERE ts BETWEEN ? AND ?", (a, b)
        ).fetchone()["n"]
        incidents_n = con.execute(
            "SELECT COUNT(*) n FROM incidents WHERE ts BETWEEN ? AND ?", (a, b)
        ).fetchone()["n"]

        vehicle_types = {
            r["vehicle_type"]: r["n"]
            for r in con.execute(
                "SELECT vehicle_type, COUNT(*) n FROM events "
                "WHERE ts BETWEEN ? AND ? AND kind='vehicle' GROUP BY vehicle_type", (a, b)
            )
        }

        latest_events = [
            dict(r) for r in con.execute(
                "SELECT ts as time,camera,vehicle_type as type,track_id,direction,confidence "
                "FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100", (a, b)
            )
        ]
        latest_plates = [
            dict(r) for r in con.execute(
                "SELECT ts as time,camera,plate,confidence,vehicle_type,track_id "
                "FROM plates WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100", (a, b)
            )
        ]
        latest_incidents = [
            dict(r) for r in con.execute(
                "SELECT ts as time,camera,type,severity,description,track_id "
                "FROM incidents WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 100", (a, b)
            )
        ]

    names = camera_name_map()
    for item in latest_events:
        item["camera_name"] = names.get(item["camera"], f"Camera {item['camera']}")
    for item in latest_plates:
        item["camera_name"] = names.get(item["camera"], f"Camera {item['camera']}")
    for item in latest_incidents:
        item["camera_name"] = names.get(item["camera"], f"Camera {item['camera']}")

    return {
        "period": period, "start": a, "end": b,
        "people_passed": people,
        "vehicles_passed": vehicles,
        "readable_plates": plates_n,
        "major_incidents": incidents_n,
        "vehicle_types": vehicle_types,
        "latest_events": latest_events,
        "latest_plates": latest_plates,
        "incidents": latest_incidents
    }


@app.get("/api/stats")
def stats():
    return history("today")


def mjpeg(worker):
    while worker.running:
        with worker.frame_lock:
            frame = None if worker.frame is None else worker.frame.copy()

        if frame is None:
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            message = f"{worker.name}: {worker.status}"
            cv2.putText(
                placeholder, message, (60, 245),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2
            )
            frame = placeholder

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.04)


@app.get("/api/cameras/{camera_id}/stream")
def stream(camera_id: int):
    with workers_lock:
        worker = workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera not found or disabled")
    return StreamingResponse(
        mjpeg(worker),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


BASE = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="dashboard")
