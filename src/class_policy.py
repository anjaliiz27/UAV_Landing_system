"""
Class policy - decided, 13 Nov 2026.
Every module must import from here. Do not keep private copies.
"""

# Always allowed for landing
SAFE_CLASSES = {1, 2, 3}          # paved-area, dirt, grass

# Allowed only when allow_conditional=True; the KG penalises them by payload
CONDITIONAL_CLASSES = {4, 8}      # gravel, vegetation

# Everything else is rejected:
# 0 unlabeled, 5 water, 6 rocks, 7 pool, 9 roof, 10 wall, 11 window, 12 door,
# 13 fence, 14 fence-pole, 15 person, 16 dog, 17 car, 18 bicycle, 19 tree,
# 20 bald-tree, 21 ar-marker, 22 obstacle, 23 conflicting

NUM_CLASSES = 24

CLASS_NAMES = {
    0: "unlabeled", 1: "paved-area", 2: "dirt", 3: "grass", 4: "gravel",
    5: "water", 6: "rocks", 7: "pool", 8: "vegetation", 9: "roof",
    10: "wall", 11: "window", 12: "door", 13: "fence", 14: "fence-pole",
    15: "person", 16: "dog", 17: "car", 18: "bicycle", 19: "tree",
    20: "bald-tree", 21: "ar-marker", 22: "obstacle", 23: "conflicting",
}
# Segmentation class -> terrain type used by the knowledge graph
CLASS_TO_TERRAIN = {
    1: "Pavement",   # paved-area
    2: "Dirt",       # dirt
    3: "Grass",      # grass
    4: "Dirt",       # gravel (loose ground, treated like dirt)
    8: "Grass",      # vegetation (soft ground, treated like grass)
}
def allowed_classes(allow_conditional=False):
    return SAFE_CLASSES | (CONDITIONAL_CLASSES if allow_conditional else set())