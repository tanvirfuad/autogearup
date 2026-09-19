import os, re, time, threading, sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")
DETECT_EVERY = int(os.getenv("DETECT_EVERY", "3"))
COUNT_LINE_Y = float(os.getenv("COUNT_LINE_Y", "0.60"))
STOPPED_SECONDS = int(os.getenv("STOPPED_SECONDS", "240"))
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").lower() == "true"
DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent / "cctv.db"))

def db():
    con = sqlite3.connect(DB_PATH)
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
        CREATE TABLE IF NOT EXISTS plates (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, camera INTEGER NOT NULL,
          plate TEXT NOT NULL, confidence REAL, vehicle_type TEXT, track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_plates_ts ON plates(ts);
        CREATE TABLE IF NOT EXISTS incidents (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, camera INTEGER NOT NULL,
          type TEXT NOT NULL, severity TEXT, description TEXT, track_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts);
        """)

def save_event(e):
    with db() as con:
        con.execute("INSERT INTO events(ts,camera,kind,vehicle_type,track_id,direction,confidence) VALUES(?,?,?,?,?,?,?)",
                    (e["time"], e["camera"], "person" if e["type"]=="person" else "vehicle", e["type"], e.get("track_id"), e.get("direction"), e.get("confidence")))

def save_plate(p):
    with db() as con:
        con.execute("INSERT INTO plates(ts,camera,plate,confidence,vehicle_type,track_id) VALUES(?,?,?,?,?,?)",
                    (p["time"], p["camera"], p["plate"], p.get("confidence"), p.get("vehicle_type"), p.get("track_id")))

def save_incident(i):
    with db() as con:
        con.execute("INSERT INTO incidents(ts,camera,type,severity,description,track_id) VALUES(?,?,?,?,?,?)",
                    (i["time"], i["camera"], i["type"], i.get("severity"), i.get("description"), i.get("track_id")))

app = FastAPI(title="AutoGearUp CCTV Gateway")

class CameraWorker:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
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
            self.model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
            print("YOLO loaded")
        except Exception as e:
            print("YOLO unavailable; live video will still work:", e)

        if ENABLE_PLATE_OCR:
            try:
                import easyocr
                self.ocr = easyocr.Reader(["en"], gpu=False)
                print("EasyOCR loaded")
            except Exception as e:
                print("Plate OCR disabled:", e)

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def open_capture(self):
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def run(self):
        self.load_models()
        while self.running:
            if not RTSP_URL:
                time.sleep(2)
                continue
            cap = self.open_capture()
            if not cap.isOpened():
                print("Unable to open RTSP stream; retrying...")
                time.sleep(3)
                continue
            while self.running:
                ok, frame = cap.read()
                if not ok:
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
            results = self.model.track(frame, persist=True, verbose=False, classes=list(allowed.keys()), tracker="bytetrack.yaml")
            if not results:
                return frame
            r = results[0]
            boxes = r.boxes
            if boxes is None:
                return frame

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.int().cpu().tolist()
            conf = boxes.conf.cpu().numpy().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls)

            for bb, c, cf, tid in zip(xyxy, cls, conf, ids):
                if c not in allowed:
                    continue
                x1, y1, x2, y2 = map(int, bb)
                name = allowed[c]
                cx, cy = (x1+x2)//2, (y1+y2)//2
                is_vehicle = name != "person"
                colour = (255, 170, 80) if is_vehicle else (80, 220, 170)
                cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2)
                cv2.putText(frame, f"{name} {cf:.0%}" + (f" ID:{tid}" if tid is not None else ""), (x1, max(18,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)

                if tid is not None:
                    side = 1 if cy >= line_y else -1
                    prev = self.prev_side.get(tid)
                    if prev is not None and prev != side:
                        if name == "person":
                            self.people_passed += 1
                        else:
                            self.vehicles_passed += 1
                            self.vehicle_types[name] += 1
                        event = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": 1, "type": name, "track_id": tid,
                            "direction": "inbound" if side == 1 else "outbound",
                            "confidence": round(float(cf), 3)
                        }
                        self.events.appendleft(event)
                        save_event(event)
                    self.prev_side[tid] = side

                    if is_vehicle:
                        last = self.last_pos.get(tid)
                        now = time.time()
                        if last:
                            dist = ((cx-last[0])**2 + (cy-last[1])**2) ** 0.5
                            if dist < 10:
                                self.still_since.setdefault(tid, now)
                                if now - self.still_since[tid] >= STOPPED_SECONDS and tid not in self.incident_latched:
                                    self.incident_latched.add(tid)
                                    incident = {
                                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "camera": 1,
                                        "type": "stopped_vehicle",
                                        "severity": "medium",
                                        "description": f"{name.title()} stationary for {STOPPED_SECONDS}s",
                                        "track_id": tid
                                    }
                                    self.incidents.appendleft(incident)
                                    save_incident(incident)
                            else:
                                self.still_since.pop(tid, None)
                        self.last_pos[tid] = (cx, cy)

                        if self.ocr and cf > 0.65 and (self.frame_no // DETECT_EVERY) % 8 == 0:
                            self.try_plate_ocr(frame, x1, y1, x2, y2, name, tid)
        except Exception as e:
            cv2.putText(frame, f"Analytics error: {type(e).__name__}", (15,35),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2)
        return frame

    def try_plate_ocr(self, frame, x1, y1, x2, y2, vehicle_type, tid):
        vh = max(1, y2-y1)
        crop = frame[y1 + vh//2:y2, x1:x2]
        if crop.size == 0:
            return
        try:
            detections = self.ocr.readtext(crop, detail=1, paragraph=False)
            for _, txt, score in detections:
                clean = re.sub(r"[^A-Z0-9]", "", txt.upper())
                if 4 <= len(clean) <= 8 and score >= 0.50:
                    if not any(p.get("plate") == clean and p.get("track_id") == tid for p in list(self.plates)[:10]):
                        self.readable_plates += 1
                        plate_event = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": 1, "plate": clean,
                            "confidence": round(float(score),3),
                            "vehicle_type": vehicle_type,
                            "track_id": tid
                        }
                        self.plates.appendleft(plate_event)
                        save_plate(plate_event)
                    break
        except Exception:
            pass

worker = CameraWorker()

@app.on_event("startup")
def startup():
    init_db()
    worker.start()

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "camera_configured": bool(RTSP_URL),
        "frame_available": worker.frame is not None,
        "analytics_enabled": worker.model is not None,
        "plate_ocr_enabled": worker.ocr is not None
    }


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
        people = con.execute("SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='person'", (a,b)).fetchone()["n"]
        vehicles = con.execute("SELECT COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='vehicle'", (a,b)).fetchone()["n"]
        plates_n = con.execute("SELECT COUNT(*) n FROM plates WHERE ts BETWEEN ? AND ?", (a,b)).fetchone()["n"]
        incidents_n = con.execute("SELECT COUNT(*) n FROM incidents WHERE ts BETWEEN ? AND ?", (a,b)).fetchone()["n"]
        vt = {r["vehicle_type"]: r["n"] for r in con.execute("SELECT vehicle_type, COUNT(*) n FROM events WHERE ts BETWEEN ? AND ? AND kind='vehicle' GROUP BY vehicle_type", (a,b))}
        latest_events = [dict(r) for r in con.execute("SELECT ts as time,camera,vehicle_type as type,track_id,direction,confidence FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 50", (a,b))]
        latest_plates = [dict(r) for r in con.execute("SELECT ts as time,camera,plate,confidence,vehicle_type,track_id FROM plates WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 50", (a,b))]
        latest_incidents = [dict(r) for r in con.execute("SELECT ts as time,camera,type,severity,description,track_id FROM incidents WHERE ts BETWEEN ? AND ? ORDER BY ts DESC LIMIT 50", (a,b))]
    return {
        "period": period, "start": a, "end": b,
        "people_passed": people, "vehicles_passed": vehicles,
        "readable_plates": plates_n, "major_incidents": incidents_n,
        "vehicle_types": vt, "latest_events": latest_events,
        "latest_plates": latest_plates, "incidents": latest_incidents
    }

@app.get("/api/stats")
def stats():
    return {
        "people_passed": worker.people_passed,
        "vehicles_passed": worker.vehicles_passed,
        "readable_plates": worker.readable_plates,
        "major_incidents": len(worker.incidents),
        "vehicle_types": dict(worker.vehicle_types),
        "latest_events": list(worker.events)[:20],
        "latest_plates": list(worker.plates)[:20],
        "incidents": list(worker.incidents)[:20],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def mjpeg():
    while True:
        with worker.frame_lock:
            frame = None if worker.frame is None else worker.frame.copy()
        if frame is None:
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...", (250,245),
                        cv2.FONT_HERSHEY_SIMPLEX, .9, (220,220,220), 2)
            frame = placeholder
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.04)

@app.get("/api/cameras/1/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

BASE = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="dashboard")
