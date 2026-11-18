"""
depth_module_v2.py - depth step for the Team Codex landing-site pipeline.

Changes vs v1 (from testing on top-down drone images):
  * MiDaS_small paints a smooth fake "ground ramp" over top-down images, so we
    remove the global trend and measure only LOCAL relief.
  * Image-border artifacts are ignored (margin).
  * Slope and roughness are split via a plane fit inside each candidate box:
        slope     = tilt of the best-fit plane
        roughness = scatter of the depth around that plane
  * Both are scaled PER IMAGE to ~[0, 1], so they are comparable with the
    distance and semantic terms in  C = k1*D + k2*R + k3*S  and thresholds
    do not depend on image size or MiDaS's arbitrary scale.

Usage (run depth + analysis ONCE per image, then query every candidate box):
    da = DepthAnalysis(image_rgb)                  # image_rgb: uint8 RGB, same
                                                   # resized image the CNN sees
    m  = da.box_metrics(corners)                   # {"slope": .., "roughness": ..}
    bad = da.is_too_steep(corners)                 # bool
    R   = m["roughness"]                           # use in the cost function

Try da = DepthAnalysis(image_rgb, size=512) to feed MiDaS a larger input.
Requires: torch, torchvision, timm, opencv-python, numpy
"""
import numpy as np
import cv2

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_model = None
_device = None


def _load_model():
    """Load MiDaS_small once and reuse it."""
    global _model, _device
    if _model is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        _model.to(_device).eval()
    return _model, _device


def estimate_depth(image_rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """
    MiDaS_small relative INVERSE depth (bigger = closer), min-max normalized to
    [0, 1], same H x W as the input. `size` = longest side fed to the network
    (256 matches MiDaS's default; try 384/512/768 and compare visually).
    """
    import torch
    import torch.nn.functional as F
    model, device = _load_model()
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    nh = max(32, int(round(h * scale / 32)) * 32)   # network needs multiples of 32
    nw = max(32, int(round(w * scale / 32)) * 32)
    img = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        pred = F.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic",
                             align_corners=False).squeeze().cpu().numpy()
    lo, hi = float(pred.min()), float(pred.max())
    return ((pred - lo) / (hi - lo + 1e-8)).astype(np.float32)


class DepthAnalysis:
    def __init__(self, image_rgb=None, size=256, depth=None,
                 detrend_sigma_frac=0.15, margin=16,
                 steep_threshold=0.5):
        """
        Pass image_rgb (runs MiDaS) or an existing `depth` array (for testing).
        detrend_sigma_frac: blur width (fraction of the long side) used to
            estimate the global trend that gets removed.
        margin: pixels at the image border ignored when computing references.
        steep_threshold: slope (scene-relative, ~0..1) above which a box is
            rejected. PLACEHOLDER - tune on several real images.
        """
        self.depth = depth if depth is not None else estimate_depth(image_rgb, size)
        h, w = self.depth.shape
        self.margin = margin
        self.steep_threshold = steep_threshold

        # 1) remove the smooth global trend -> local relief only
        sigma = detrend_sigma_frac * max(h, w)
        trend = cv2.GaussianBlur(self.depth, (0, 0), sigma)
        self.local = (self.depth - trend).astype(np.float32)

        # 2) per-pixel gradient of the local relief (border zeroed)
        smooth = cv2.GaussianBlur(self.local, (5, 5), 0)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        self.grad = np.sqrt(gx ** 2 + gy ** 2)
        m = margin
        self.valid = np.zeros((h, w), dtype=bool)
        self.valid[m:h - m, m:w - m] = True
        self.grad[~self.valid] = 0.0

        # 3) per-image reference scales (so outputs are ~[0, 1])
        self.grad_ref = float(np.percentile(self.grad[self.valid], 99)) + 1e-8
        self.relief_ref = float(np.percentile(np.abs(self.local[self.valid]), 99)) + 1e-8

    # ---- helpers -------------------------------------------------------
    def _box_xyz(self, corners):
        corners = np.asarray(corners, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(corners)
        x0, y0 = max(x, 0), max(y, 0)
        x1 = min(x + w, self.local.shape[1])
        y1 = min(y + h, self.local.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(mask, [corners - np.array([x0, y0], dtype=np.int32)], 1)
        ys, xs = np.nonzero(mask)
        if xs.size < 10:
            return None
        z = self.local[y0:y1, x0:x1][ys, xs]
        return xs.astype(np.float32), ys.astype(np.float32), z

    # ---- public API ----------------------------------------------------
    def box_metrics(self, corners) -> dict:
        """
        Fit a plane z = a*x + b*y + c to the local relief inside the box.
        slope     = sqrt(a^2 + b^2) / grad_ref      (tilt, scene-relative)
        roughness = std(residuals)  / relief_ref    (bumpiness, scene-relative)
        Both clipped to [0, 1]. Unusable boxes return inf (treat as invalid).
        """
        data = self._box_xyz(corners)
        if data is None:
            return {"slope": float("inf"), "roughness": float("inf")}
        xs, ys, z = data
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ coef
        slope = np.hypot(coef[0], coef[1]) / self.grad_ref
        rough = resid.std() / self.relief_ref
        return {"slope": float(min(slope, 1.0)), "roughness": float(min(rough, 1.0))}

    def is_too_steep(self, corners) -> bool:
        """Reject a candidate whose plane tilt exceeds the threshold."""
        m = self.box_metrics(corners)
        return bool(m["slope"] > self.steep_threshold)


def box_corners(cx, cy, w, h, angle_deg=0.0):
    """4 corners of a rotated rectangle (helper for testing)."""
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
    return cv2.boxPoints(rect)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    bgr = cv2.imread(path)
    assert bgr is not None, f"could not read {path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (800, 600))            # same size the rest of the team uses

    da = DepthAnalysis(rgb, size=size)

    def vis(a):
        a = (255 * (a - a.min()) / (a.max() - a.min() + 1e-8)).astype(np.uint8)
        return cv2.applyColorMap(a, cv2.COLORMAP_INFERNO)

    cv2.imwrite("v2_depth.png", vis(da.depth))
    cv2.imwrite("v2_local_relief.png", vis(da.local))   # ramp removed
    cv2.imwrite("v2_gradient.png", vis(da.grad))
    print("input size:", size, "| grad_ref:", da.grad_ref, "| relief_ref:", da.relief_ref)
    for name, (x, y) in {"centre": (400, 300), "top-left": (150, 120),
                         "bottom-right": (650, 480)}.items():
        print(name, da.box_metrics(box_corners(x, y, 80, 80)))
