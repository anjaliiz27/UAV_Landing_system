"""
3-layer spreading-activation knowledge graph.
Layer 1 (sources) -> Layer 2 (properties) -> Layer 3 (terrains).
Penalties come only from edge-weight accumulation (matrix products).
No if/else on traits or terrains, and no hardcoded penalty values.
"""
import json
import numpy as np
import cv2
from class_policy import CLASS_TO_TERRAIN, NUM_CLASSES

SOURCES  = ["Fragile", "Valuable", "Biohazard", "Heavy"]
PROPS    = ["Hard", "Wet", "Slippery", "Dirty", "Visible", "Contaminated", "Soft", "Unstable"]
TERRAINS = ["Pavement", "Grass", "Dirt"]

# Segmentation class ID -> terrain index (Pavement=0, Grass=1, Dirt=2).
# Classes with no terrain stay at -1 and are ignored when finding the dominant terrain.
_LUT = np.full(NUM_CLASSES, -1, dtype=np.int32)
for _class_id, _terrain in CLASS_TO_TERRAIN.items():
    _LUT[_class_id] = TERRAINS.index(_terrain)

# Layer 1 -> Layer 2: how much a package trait cares about (is hurt by) a property.
# Weights are team-designed design choices; tune and justify them in the report.
E1 = {
    ("Fragile", "Hard"): 0.9,  ("Fragile", "Unstable"): 0.5,
    ("Fragile", "Slippery"): 0.3,  ("Fragile", "Wet"): 0.2,
    ("Valuable", "Visible"): 0.8,  ("Valuable", "Unstable"): 0.4,
    ("Valuable", "Wet"): 0.3,  ("Valuable", "Dirty"): 0.3,  ("Valuable", "Hard"): 0.3,
    ("Biohazard", "Contaminated"): 0.9,  ("Biohazard", "Wet"): 0.6,
    ("Biohazard", "Dirty"): 0.5,  ("Biohazard", "Slippery"): 0.2,
    ("Heavy", "Soft"): 0.8,  ("Heavy", "Unstable"): 0.8,  ("Heavy", "Slippery"): 0.3,
}

# Layer 2 -> Layer 3: how strongly a terrain exhibits a property.
E2 = {
    ("Hard", "Pavement"): 0.9,  ("Hard", "Grass"): 0.1,  ("Hard", "Dirt"): 0.4,
    ("Wet", "Pavement"): 0.1,  ("Wet", "Grass"): 0.5,  ("Wet", "Dirt"): 0.4,
    ("Slippery", "Pavement"): 0.5,  ("Slippery", "Grass"): 0.4,  ("Slippery", "Dirt"): 0.3,
    ("Dirty", "Pavement"): 0.1,  ("Dirty", "Grass"): 0.4,  ("Dirty", "Dirt"): 0.9,
    ("Visible", "Pavement"): 0.8,  ("Visible", "Grass"): 0.3,  ("Visible", "Dirt"): 0.4,
    ("Contaminated", "Pavement"): 0.2,  ("Contaminated", "Grass"): 0.4,  ("Contaminated", "Dirt"): 0.8,
    ("Soft", "Pavement"): 0.0,  ("Soft", "Grass"): 0.6,  ("Soft", "Dirt"): 0.8,
    ("Unstable", "Pavement"): 0.0,  ("Unstable", "Grass"): 0.3,  ("Unstable", "Dirt"): 0.7,
}


def _matrix(edges, rows, cols):
    M = np.zeros((len(rows), len(cols)), np.float32)
    for (r, c), w in edges.items():
        M[rows.index(r), cols.index(c)] = w
    return M


W1 = _matrix(E1, SOURCES, PROPS)       # 4 x 8
W2 = _matrix(E2, PROPS, TERRAINS)      # 8 x 3

# Scale from the graph itself: the largest penalty when every source is active.
_SCALE = float((np.ones(len(SOURCES), np.float32) @ W1 @ W2).max())


def load_active_nodes(path_or_dict):
    """Read mission_config.json (or a dict). Returns the 0/1 source-node vector."""
    cfg = path_or_dict
    if not isinstance(cfg, dict):
        with open(path_or_dict) as f:
            cfg = json.load(f)
    pkg = cfg.get("package", cfg)
    return np.array([float(pkg.get(s.lower()) is True) for s in SOURCES], np.float32)


def spread(active):
    """Spreading activation. Returns (terrain_penalty[3] in 0-1, trace)."""
    a1 = np.asarray(active, np.float32)
    a2 = a1 @ W1                       # property activation
    a3 = a2 @ W2                       # raw terrain penalty
    pen = a3 / _SCALE
    trace = {
        "active_sources": [s for s, v in zip(SOURCES, a1) if v > 0],
        "property_activation": {p: round(float(v), 3) for p, v in zip(PROPS, a2) if v > 0},
        "terrain_penalty": {t: round(float(v), 3) for t, v in zip(TERRAINS, pen)},
    }
    return pen, trace


def dominant_terrain(class_map, corners, step=4):
    """
    Sample interior points of the box; return the index of the most frequent
    terrain, or None if no sampled pixel belongs to a mapped terrain.
    """
    H, W = class_map.shape
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.asarray(corners).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask[::step, ::step])
    terr = _LUT[class_map[ys * step, xs * step]]
    terr = terr[terr >= 0]                         # ignore pixels whose class has no terrain
    if terr.size == 0:
        return None
    return int(np.argmax(np.bincount(terr, minlength=len(TERRAINS))))


def semantic_penalty(class_map, corners, active):
    """S for one candidate box. Returns (S, terrain_name, trace)."""
    pen, trace = spread(active)
    t = dominant_terrain(class_map, corners)
    if t is None:
        # No mapped terrain was sampled: be cautious and use the worst terrain.
        t = int(np.argmax(pen))
        trace["note"] = "no mapped terrain sampled; worst-case terrain used"
    trace["dominant_terrain"] = TERRAINS[t]
    return float(pen[t]), TERRAINS[t], trace