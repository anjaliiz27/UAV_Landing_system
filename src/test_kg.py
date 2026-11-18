
from itertools import combinations

from semantic_brain import load_active_nodes, spread

traits = ["fragile", "valuable", "biohazard", "heavy"]

print(f"{'Package configuration':<45} {'Pavement':>10} {'Grass':>10} {'Dirt':>10}")
print("-" * 80)

for r in range(1, len(traits) + 1):
    for combo in combinations(traits, r):

        config = {trait: True for trait in combo}

        active = load_active_nodes({"package": config})
        penalties, trace = spread(active)

        values = trace["terrain_penalty"]

        name = " + ".join(combo)

        print(
            f"{name:<45} "
            f"{values['Pavement']:>10.3f} "
            f"{values['Grass']:>10.3f} "
            f"{values['Dirt']:>10.3f}"
        )
