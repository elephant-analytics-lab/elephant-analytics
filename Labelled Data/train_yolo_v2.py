import os
import zipfile
import shutil
from ultralytics import YOLO
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

P5_ZIP = os.path.join(BASE_DIR, "project-5-at-2025-11-15-13-59-7db52ff8.zip")
E10_ZIP = os.path.join(BASE_DIR, "elephants10.zip")

P5_DIR = os.path.join(DATASETS_DIR, "project5")
E10_DIR = os.path.join(DATASETS_DIR, "elephants10")
COMB_DIR = os.path.join(DATASETS_DIR, "elephants_combined")

os.makedirs(DATASETS_DIR, exist_ok=True)

def ensure_unzip(zip_path, target_dir):
    if not os.path.exists(target_dir):
        print(f"Unzipping {os.path.basename(zip_path)} to {target_dir} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_dir)
    else:
        print(f"{target_dir} already exists, skipping unzip")

def find_dir_with_ext(base_dir, exts):
    """Find first directory under base_dir that contains files with given extensions."""
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(exts) for f in files):
            return root
    return None

# 1) unzip both datasets
ensure_unzip(P5_ZIP, P5_DIR)
ensure_unzip(E10_ZIP, E10_DIR)

# 2) locate images/labels in project5
p5_images = os.path.join(P5_DIR, "images")
p5_labels = os.path.join(P5_DIR, "labels")

print("project5 images:", p5_images)
print("project5 labels:", p5_labels)

# 3) locate images/labels in elephants10 (auto-detect)
e10_images = find_dir_with_ext(E10_DIR, (".jpg", ".jpeg", ".png", ".bmp"))
e10_labels = find_dir_with_ext(E10_DIR, (".txt",))

print("elephants10 root:", E10_DIR)
print("elephants10 images dir:", e10_images)
print("elephants10 labels dir:", e10_labels)

if e10_images is None or e10_labels is None:
    raise RuntimeError("Could not find images or labels folder inside elephants10 dataset")

# 4) build combined dataset
comb_images = os.path.join(COMB_DIR, "images")
comb_labels = os.path.join(COMB_DIR, "labels")
os.makedirs(comb_images, exist_ok=True)
os.makedirs(comb_labels, exist_ok=True)

def copy_dir(src, dst, tag):
    for name in os.listdir(src):
        src_f = os.path.join(src, name)
        dst_f = os.path.join(dst, name)
        if not os.path.isfile(src_f):
            continue
        if os.path.exists(dst_f):
            print(f"[{tag}] skip duplicate {name}")
            continue
        shutil.copy2(src_f, dst_f)

copy_dir(p5_images, comb_images, "p5-img")
copy_dir(p5_labels, comb_labels, "p5-lab")
copy_dir(e10_images, comb_images, "e10-img")
copy_dir(e10_labels, comb_labels, "e10-lab")

print("combined images:", len(os.listdir(comb_images)))
print("combined labels:", len(os.listdir(comb_labels)))

# 5) write data.yaml for combined dataset
yaml_text = f"""
path: {COMB_DIR}
train: images
val: images

names:
  0: adult
  1: baby
"""

yaml_path = os.path.join(COMB_DIR, "elephant_combined.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("Wrote data.yaml to:", yaml_path)
print(open(yaml_path).read())

# 6) train YOLO
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=40,
    imgsz=640,
    batch=16,
    device=device,
    name="elephant_combined_v2",
)

print("Training finished. Check runs/detect/elephant_combined_v2/weights for best.pt")
