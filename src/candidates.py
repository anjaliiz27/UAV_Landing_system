import cv2
import numpy as np

from depth_module_v2 import box_corners
from class_policy import allowed_classes, CONDITIONAL_CLASSES, NUM_CLASSES


def _box_kernel(w, h, angle):
    """Filled, rotated rectangle normalised to sum 1 (gives the safe fraction)."""
    k = int(np.ceil(np.hypot(w, h))) + 3
    if k % 2 == 0:
        k += 1
    c = k // 2
    pts = box_corners(c, c, w, h, angle).astype(np.int32)
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.fillPoly(kernel, [pts], 1.0)
    total = kernel.sum()
    if total == 0:
        raise ValueError("Invalid box kernel.")
    return kernel / total


def _rotated_iou(box1, box2):
    r1 = ((float(box1[0]), float(box1[1])), (float(box1[2]), float(box1[3])), float(box1[4]))
    r2 = ((float(box2[0]), float(box2[1])), (float(box2[2]), float(box2[3])), float(box2[4]))
    _, inter = cv2.rotatedRectangleIntersection(r1, r2)
    if inter is None:
        return 0.0
    inter_area = abs(cv2.contourArea(inter))
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / union if union > 0 else 0.0


def generate_candidates(
    class_map,
    target,
    da,
    box_w=60,
    box_h=40,
    angles=range(0, 180, 15),
    min_safe_frac=0.995,
    stride=8,
    max_depth_checks=300,
    top_k=50,
    nms_iou=0.0,
    allow_conditional=False,
    erode_px=15,              # safety margin around non-safe areas (odd number of pixels)
    probs=None,               # optional (NUM_CLASSES, H, W) softmax output from predict()
    prob_safe_min=0.8,        # used only when probs is given: min P(safe) per pixel
):
    """
    Generate and validate rotatable landing boxes.

    Returns {"status", "candidates", "n_rejected"}.
    status is "ok", "no_valid_candidate" or "invalid_target".
    """
    # ---------- input validation ----------
    class_map = np.asarray(class_map)
    if class_map.ndim != 2:
        raise ValueError("class_map must be a 2D array.")
    if not np.issubdtype(class_map.dtype, np.integer):
        raise ValueError("class_map must contain integer class IDs.")
    H, W = class_map.shape
    if da.depth.shape != (H, W):
        raise ValueError("Semantic map and MiDaS depth dimensions must match.")
    if box_w <= 0 or box_h <= 0:
        raise ValueError("Package dimensions must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not 0 < min_safe_frac <= 1:
        raise ValueError("min_safe_frac must be between 0 and 1.")
    if not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou must be between 0 and 1.")
    if erode_px < 1:
        raise ValueError("erode_px must be at least 1.")
    if class_map.min() < 0 or class_map.max() >= NUM_CLASSES:
        raise ValueError(f"Class IDs must be between 0 and {NUM_CLASSES - 1}.")
    if probs is not None and probs.shape != (NUM_CLASSES, H, W):
        raise ValueError(f"probs must have shape ({NUM_CLASSES}, {H}, {W}).")

    tx, ty = map(float, target)
    if not (0 <= tx < W and 0 <= ty < H):
        return {"status": "invalid_target", "candidates": [], "n_rejected": 0}

    # ---------- safe-zone mask ----------
    allowed = sorted(allowed_classes(allow_conditional))
    safe_bool = np.isin(class_map, allowed)
    if probs is not None:
        # the model must also be confident that the pixel is safe
        p_safe = probs[allowed].sum(axis=0)
        safe_bool &= (p_safe >= prob_safe_min)
    safe = safe_bool.astype(np.float32)
    safe = cv2.erode(safe, np.ones((erode_px, erode_px), np.uint8))
    diag = float(np.hypot(H, W))

    # ---------- step 1: rotational search ----------
    raw = []
    for angle in angles:
        kernel = _box_kernel(box_w, box_h, angle)
        frac = cv2.filter2D(safe, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        sampled = frac[::stride, ::stride]
        ys, xs = np.nonzero(sampled >= min_safe_frac)
        for y, x in zip(ys * stride, xs * stride):
            d = np.hypot(x - tx, y - ty) / diag
            raw.append((float(d), int(x), int(y), float(angle)))

    if not raw:
        return {"status": "no_valid_candidate", "candidates": [], "n_rejected": 0}

    # ---------- step 2: nearest first ----------
    raw.sort(key=lambda item: item[0])

    # ---------- step 3: rotated-box NMS (compare only nearby boxes) ----------
    kept = []
    kx = np.zeros(max_depth_checks, np.float32)
    ky = np.zeros(max_depth_checks, np.float32)
    reach = float(np.hypot(box_w, box_h))

    for d, x, y, angle in raw:
        n = len(kept)
        overlaps = False
        if n:
            near = np.nonzero(np.hypot(kx[:n] - x, ky[:n] - y) < reach)[0]
            for j in near:
                prev = kept[j]
                if _rotated_iou((x, y, box_w, box_h, angle),
                                (prev[1], prev[2], box_w, box_h, prev[3])) > nms_iou:
                    overlaps = True
                    break
        if not overlaps:
            kx[n], ky[n] = x, y
            kept.append((d, x, y, angle))
            if len(kept) >= max_depth_checks:
                break

    # ---------- step 4: depth validation + class histogram ----------
    output, rejected = [], 0
    cond_ids = sorted(CONDITIONAL_CLASSES)

    for d, x, y, angle in kept:
        corners = box_corners(x, y, box_w, box_h, angle)
        m = da.box_metrics(corners)
        slope, rough = m["slope"], m["roughness"]

        if not np.isfinite(slope) or not np.isfinite(rough):
            rejected += 1
            continue
        if slope > da.steep_threshold:
            rejected += 1
            continue

        region = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(region, [corners.astype(np.int32)], 1)
        values = class_map[region == 1]
        if values.size == 0:
            rejected += 1
            continue

        hist = np.bincount(values, minlength=NUM_CLASSES).astype(np.float32)
        hist /= hist.sum()

        output.append({
            "x": int(x), "y": int(y),
            "w": int(box_w), "h": int(box_h),
            "angle": float(angle),
            "corners": corners,
            "D": float(d),
            "R": float(rough),
            "slope": float(slope),
            "class_hist": hist,
            "conditional_frac": float(hist[cond_ids].sum()),
        })

    output.sort(key=lambda c: c["D"])
    if top_k is not None:
        output = output[:top_k]

    return {
        "status": "ok" if output else "no_valid_candidate",
        "candidates": output,
        "n_rejected": rejected,
    }