import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

source = os.getenv("TENSORRT_SOURCE_MODEL", "yolo26s.pt")
imgsz = int(os.getenv("DETECTION_IMGSZ", "640"))
batch = int(os.getenv("ANALYTICS_BATCH_SIZE", "8"))
device = os.getenv("TENSORRT_DEVICE", "0")
workspace = float(os.getenv("TENSORRT_WORKSPACE_GB", "4"))

print("AutoGearUp TensorRT export")
print(f"Source model : {source}")
print(f"Image size   : {imgsz}")
print(f"Max batch    : {batch}")
print(f"Device       : {device}")
print()
print("TensorRT engines are hardware/runtime specific. Export on the NVIDIA")
print("machine that will actually run the CCTV analytics gateway.")
print()

model = YOLO(source)
output = model.export(
    format="engine",
    imgsz=imgsz,
    batch=batch,
    dynamic=True,
    quantize=16,
    device=device,
    workspace=workspace,
)

output = Path(output)
print()
print(f"Created: {output}")
print("Set this in .env and restart the gateway:")
print(f"YOLO_MODEL={output}")
