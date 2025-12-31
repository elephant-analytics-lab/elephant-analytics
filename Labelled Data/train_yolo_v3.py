import os
import zipfile
import shutil
from ultralytics import YOLO
import torch

# ------------ Base paths ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # this Labelled Data folder
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
COMB_DIR = os.path.join(DATASETS_DIR, "elephants_all_v3")

os.makedirs(DATASETS_DIR, exist_ok=True)

# Three datasets (exact filenames from your screenshot)
DATASETS = [
    {
        "name": "project5",
        "zip": "project-5-at-2025-11-15-13-59-7db52ff8.zip",
        "yolo_layout": True,   # already has images/ + labels/
    },
    {
        "name": "elephants10",
        "zip": "elephants10.zip",
        "yolo_layout": False,  # auto-detect folders
    },
    {
        "name": "elephant_calfs",
        "zip": "Elephant and Calfs.zip",
        "yolo_layout": False,  # auto-detect folders
    },
]


def ensure_unzip(zip_path, target_dir):
    """Unzip zip_path into target_dir once."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    if not os.path.exists(target_dir):
        print(f"[UNZIP] {os.path.basename(zip_path)} -> {target_dir}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_dir)
    else:
        print(f"[SKIP] {target_dir} already exists")


def find_dir_with_ext(base_dir, exts):
    """Find first directory under base_dir that contains files with given extensions."""
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(exts) for f in files):
            return root
    return None


# ------------ Step 1: unzip all 3 datasets ------------
dataset_infos = []

for ds in DATASETS:
    zip_path = os.path.join(BASE_DIR, ds["zip"])
    out_dir = os.path.join(DATASETS_DIR, ds["name"])

    ensure_unzip(zip_path, out_dir)

    if ds["yolo_layout"]:
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
    else:
        img_dir = find_dir_with_ext(out_dir, (".jpg", ".jpeg", ".png", ".bmp"))
        lbl_dir = find_dir_with_ext(out_dir, (".txt",))

    print(f"[DATASET] {ds['name']}")
    print("  root   :", out_dir)
    print("  images :", img_dir)
    print("  labels :", lbl_dir)

    if img_dir is None or lbl_dir is None:
        raise RuntimeError(f"Could not find images/labels for dataset {ds['name']}")

    dataset_infos.append((img_dir, lbl_dir, ds["name"]))

# ------------ Step 2: build combined dataset ------------
# start fresh each time so it doesn't accumulate old stuff
if os.path.exists(COMB_DIR):
    print(f"[CLEAN] removing old combined dir {COMB_DIR}")
    shutil.rmtree(COMB_DIR)

comb_images = os.path.join(COMB_DIR, "images")
comb_labels = os.path.join(COMB_DIR, "labels")
os.makedirs(comb_images, exist_ok=True)
os.makedirs(comb_labels, exist_ok=True)


def copy_dir(src, dst, tag):
    count = 0
    for name in os.listdir(src):
        src_f = os.path.join(src, name)
        dst_f = os.path.join(dst, name)
        if not os.path.isfile(src_f):
            continue
        if os.path.exists(dst_f):
            print(f"[{tag}] skip duplicate {name}")
            continue
        shutil.copy2(src_f, dst_f)
        count += 1
    print(f"[{tag}] copied {count} files")


for img_dir, lbl_dir, name in dataset_infos:
    copy_dir(img_dir, comb_images, f"{name}-img")
    copy_dir(lbl_dir, comb_labels, f"{name}-lbl")

print("Combined images:", len(os.listdir(comb_images)))
print("Combined labels:", len(os.listdir(comb_labels)))

# ------------ Step 3: write data.yaml ------------
yaml_text = f"""
path: {COMB_DIR}
train: images
val: images

names:
  0: adult
  1: baby
"""

yaml_path = os.path.join(COMB_DIR, "elephant_all_v3.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("Wrote data.yaml to:", yaml_path)
print(open(yaml_path).read())

# ------------ Step 4: train YOLO ------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# use the base model you already have in this folder
MODEL_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")
if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"Cannot find {MODEL_WEIGHTS} (download yolov8n.pt here)")

model = YOLO(MODEL_WEIGHTS)

results = model.train(
    data=yaml_path,
    epochs=50,          # more epochs for more data (adjust if too slow)
    imgsz=640,
    batch=16,
    device=device,
    name="elephant_all_v3",
)

print("Training finished. Check runs/detect/elephant_all_v3/weights/best.pt")
