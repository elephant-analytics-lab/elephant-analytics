import os
from ultralytics import YOLO

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this folder (Labelled Data)
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
PROJECT5_ZIP = os.path.join(
    BASE_DIR,
    "project-5-at-2025-11-15-13-59-7db52ff8.zip"  # zip name in this folder
)

os.makedirs(DATASETS_DIR, exist_ok=True)

# === Unzip project-5 dataset ===
project5_dir = os.path.join(DATASETS_DIR, "project5")
images_dir = os.path.join(project5_dir, "images")

if not os.path.exists(images_dir):
    import zipfile
    print("Unzipping project-5 dataset...")
    with zipfile.ZipFile(PROJECT5_ZIP) as z:
        z.extractall(project5_dir)
else:
    print("Project5 dataset already unzipped.")

print("Project5 contents:", os.listdir(project5_dir))

# === Write YOLO data.yaml ===
yaml_text = f"""
path: {project5_dir}
train: images
val: images

names:
  0: adult
  1: baby
"""

yaml_path = os.path.join(project5_dir, "elephant_project5.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("Wrote data.yaml to:", yaml_path)
print(open(yaml_path).read())

# === Check torch / GPU ===
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# === Train YOLO ===
model = YOLO("yolov8n.pt")  # small model to start with

results = model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,        # reduce to 8 or 4 if OOM
    device=device,
    name="elephant_project5_v1",
)

print("Training finished. Check runs/detect/elephant_project5_v1/weights for best.pt")
