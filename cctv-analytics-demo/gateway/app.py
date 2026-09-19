import os, re, time, threading
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
                        self.events.appendleft({
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": 1, "type": name, "track_id": tid,
                            "direction": "inbound" if side == 1 else "outbound",
                            "confidence": round(float(cf), 3)
                        })
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
                                    self.incidents.appendleft({
                                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "camera": 1,
                                        "type": "stopped_vehicle",
                                        "severity": "medium",
                                        "description": f"{name.title()} stationary for {STOPPED_SECONDS}s",
                                        "track_id": tid
                                    })
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
                        self.plates.appendleft({
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera": 1, "plate": clean,
                            "confidence": round(float(score),3),
                            "vehicle_type": vehicle_type,
                            "track_id": tid
                        })
                    break
        except Exception:
            pass

worker = CameraWorker()

@app.on_event("startup")
def startup():
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
