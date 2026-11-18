import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from candidates import generate_candidates
from depth_module_v2 import DepthAnalysis
from class_policy import CLASS_NAMES

# ---------- configuration ----------
NAME = sys.argv[1] if len(sys.argv) > 1 else "179"     # python test_candidates.py 489
W, H = 800, 600
BOX_W, BOX_H = 60, 40
TARGET = (400, 300)
SHOW_PLOTS = True

root = Path(__file__).resolve().parent
image_path = root / "test_images" / f"{NAME}.jpg"
mask_path = root / "test_masks" / f"{NAME}.png"

# ---------- load image ----------
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")
if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

bgr = cv2.imread(str(image_path))
if bgr is None:
    raise ValueError("Could not read input image.")
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

# ---------- load class-ID mask ----------
class_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
if class_map is None:
    raise ValueError("Could not read semantic mask.")
if class_map.ndim != 2:
    raise ValueError("Expected a single-channel class-ID mask (use masks_800x600).")
if class_map.shape != (H, W):
    class_map = cv2.resize(class_map, (W, H), interpolation=cv2.INTER_NEAREST)
class_map = class_map.astype(np.int32)
print("Class IDs in mask:", np.unique(class_map))

# ---------- depth (MiDaS runs once) ----------
print("Running MiDaS depth analysis...")
da = DepthAnalysis(img)
print("Depth shape:", da.depth.shape,
      "| grad_ref:", round(da.grad_ref, 5),
      "| relief_ref:", round(da.relief_ref, 5))


def show(result, mode):
    cands = result["candidates"]
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.plot(*TARGET, "r*", markersize=18, label="Target")
    for i, c in enumerate(cands[:10]):
        poly = np.vstack([c["corners"], c["corners"][0]])
        plt.plot(poly[:, 0], poly[:, 1], "lime" if i == 0 else "yellow", linewidth=2)
        plt.text(c["x"], c["y"], str(i + 1), color="white",
                 fontsize=10, ha="center", va="center")
    tag = "conditional ON" if mode else "strict"
    plt.title(f"Landing candidates - {NAME} ({tag})")
    plt.legend()
    plt.axis("off")
    out = root / f"candidates_{NAME}_{'cond' if mode else 'strict'}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("Saved:", out)


# ---------- run both modes ----------
for mode in (False, True):
    print("\n" + "=" * 60)
    print("allow_conditional =", mode)
    result = generate_candidates(
        class_map=class_map, target=TARGET, da=da,
        box_w=BOX_W, box_h=BOX_H,
        min_safe_frac=0.995, stride=16,
        max_depth_checks=300, top_k=50,
        nms_iou=0.0, allow_conditional=mode,
    )
    cands = result["candidates"]
    print("Status:", result["status"],
          "| valid:", len(cands),
          "| rejected by depth:", result["n_rejected"])

    for i, c in enumerate(cands[:10]):
        top = np.argsort(c["class_hist"])[::-1][:3]
        mix = {CLASS_NAMES[int(k)]: round(float(c["class_hist"][k]), 2)
               for k in top if c["class_hist"][k] > 0.01}
        print(f"  #{i + 1} centre=({c['x']},{c['y']}) angle={c['angle']:.0f} "
              f"D={c['D']:.3f} R={c['R']:.3f} slope={c['slope']:.3f} "
              f"cond={c['conditional_frac']:.2f} classes={mix}")

    show(result, mode)

# ---------- failure-case checks ----------
bad = generate_candidates(class_map, (W + 50, H + 50), da)
print("\nTarget outside image ->", bad["status"])

if SHOW_PLOTS:
    plt.show()

