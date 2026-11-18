"""Dataset inspection + preprocessing + shared constants.

CLI (run from repo root):
    python -m src.preprocessing check    # inspect dataset, find NUM_CLASSES
    python -m src.preprocessing prep     # write 800x600 copies to PREP_DIR
"""
import os, sys, glob
import cv2
import numpy as np

# ---------------- shared constants (single source of truth) ----------------
DATA_ROOT = os.environ.get("DATA_ROOT", "./semantic_drone_dataset")
PREP_DIR = os.environ.get("PREP_DIR", "./prepared_800x600")
MAIN_W, MAIN_H = 800, 600        # canonical "main image" resolution
NET_SIZE = 256                   # network resolution
NUM_CLASSES = 24                 # set to max label id + 1 after running `check`
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_dirs(root=DATA_ROOT):
    img_dir = lbl_dir = None
    for d, _, _ in os.walk(root):
        name = os.path.basename(d)
        if name == "original_images":
            img_dir = d
        elif name == "label_images_semantic":
            lbl_dir = d
    assert img_dir and lbl_dir, f"original_images/ or label_images_semantic/ not found under {root}"
    return img_dir, lbl_dir


def check_dataset():
    img_dir, lbl_dir = find_dirs()
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.png")))
    print(f"images: {len(imgs)}  labels: {len(lbls)}")
    m = cv2.imread(lbls[0], cv2.IMREAD_UNCHANGED)
    print("mask shape/dtype:", m.shape, m.dtype, "(must be 2-D index image)")
    counts = np.zeros(256, dtype=np.int64)
    for p in lbls:
        counts += np.bincount(cv2.imread(p, cv2.IMREAD_UNCHANGED).ravel(), minlength=256)[:256]
    present = np.nonzero(counts)[0]
    print("label ids present:", present.tolist())
    print(f"=> set NUM_CLASSES = {present.max() + 1} in src/preprocessing.py")
    for c in present:
        print(f"  class {c:2d}: {100 * counts[c] / counts.sum():6.2f}%")


# ---------------- 800x600 canonical preprocessing ----------------
def to_main(image_bgr):
    """Any input image -> canonical 800x600 main image."""
    return cv2.resize(image_bgr, (MAIN_W, MAIN_H), interpolation=cv2.INTER_AREA)


def to_net_size(main_bgr):
    """800x600 main image -> 256x256 (still BGR uint8)."""
    return cv2.resize(main_bgr, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)


def normalize(img256_bgr):
    """256x256 BGR uint8 -> float32 CxHxW, RGB, ImageNet-normalized."""
    x = (img256_bgr[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def mask_to_net_size(mask):
    return cv2.resize(mask, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_NEAREST)


def upscale_mask(mask256):
    """256x256 class mask -> 800x600 (nearest: class ids must never be blended)."""
    return cv2.resize(mask256, (MAIN_W, MAIN_H), interpolation=cv2.INTER_NEAREST)


def prepare():
    img_dir, lbl_dir = find_dirs()
    os.makedirs(f"{PREP_DIR}/images", exist_ok=True)
    os.makedirs(f"{PREP_DIR}/masks", exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + ".png")
        if not os.path.exists(lp):
            continue
        img = to_main(cv2.imread(p))
        msk = cv2.resize(cv2.imread(lp, cv2.IMREAD_UNCHANGED), (MAIN_W, MAIN_H),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{PREP_DIR}/images/{stem}.png", img)
        cv2.imwrite(f"{PREP_DIR}/masks/{stem}.png", msk)
        n += 1
    print(f"prepared {n} image/mask pairs -> {PREP_DIR}")


if __name__ == "__main__":
    {"check": check_dataset, "prep": prepare}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
