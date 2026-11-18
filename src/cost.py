
import numpy as np
from semantic_brain import semantic_penalty

# Assignment reference weights: distance, roughness, semantic
DEFAULT_K = (0.4, 0.2, 0.4)


def rank_candidates(cands, class_map, active, k=DEFAULT_K):
    """
    Rank landing candidates using:

        C = k1*D + k2*R + k3*S

    cands:
        generate_candidates()["candidates"]

    class_map:
        H x W semantic class map

    active:
        Source-node activation vector from load_active_nodes()

    k:
        (k1, k2, k3), non-negative cost weights
    """

    # Validate weights
    if len(k) != 3:
        raise ValueError("k must contain exactly three weights.")

    if not all(np.isfinite(v) and v >= 0 for v in k):
        raise ValueError("Cost weights must be finite and non-negative.")

    if not np.isclose(sum(k), 1.0):
        raise ValueError("Cost weights must sum to 1.")

    # Handle empty candidate list
    if not cands:
        return {
            "status": "no_valid_candidate",
            "ranked": [],
            "winner": None
        }

    # Validate candidate fields
    required = {"D", "R", "corners", "x", "y", "angle"}

    for i, c in enumerate(cands):
        missing = required - c.keys()
        if missing:
            raise KeyError(
                f"Candidate {i} is missing fields: {missing}"
            )

    k1, k2, k3 = k

    # Normalize distance and roughness within this candidate set
    d_max = max(float(c["D"]) for c in cands)
    r_max = max(float(c["R"]) for c in cands)

    d_max = d_max if d_max > 0 else 1.0
    r_max = r_max if r_max > 0 else 1.0

    ranked = []

    for c in cands:

        D = float(c["D"]) / d_max
        R = float(c["R"]) / r_max

        # Calculate semantic penalty using the knowledge graph
        S, terrain, trace = semantic_penalty(
            class_map,
            c["corners"],
            active
        )

        # Individual cost contributions
        distance_contribution = k1 * D
        roughness_contribution = k2 * R
        semantic_contribution = k3 * S

        # Total cost
        cost = (
            distance_contribution
            + roughness_contribution
            + semantic_contribution
        )

        ranked.append({
            **c,
            "D_norm": D,
            "R_norm": R,
            "S": S,
            "terrain": terrain,
            "cost": float(cost),
            "contrib": {
                "distance": distance_contribution,
                "roughness": roughness_contribution,
                "semantic": semantic_contribution
            },
            "kg_trace": trace
        })

    # Lowest cost is the best candidate
    ranked.sort(key=lambda candidate: candidate["cost"])

    # Assign ranks after sorting
    for i, candidate in enumerate(ranked):
        candidate["rank"] = i + 1

    return {
        "status": "ok",
        "ranked": ranked,
        "winner": ranked[0]
    }


def explain(result, top=3):
    """Generate a human-readable explanation of the ranking."""

    if result["status"] != "ok":
        return "No valid landing site found."

    w = result["winner"]

    lines = [
        f"Winner at ({w['x']}, {w['y']}), "
        f"angle {w['angle']:.0f} degrees",
        f"Terrain: {w['terrain']}",
        f"Total cost: {w['cost']:.4f}",
        "",
        "Cost breakdown:",
        f"Distance contribution: {w['contrib']['distance']:.4f}",
        f"Roughness contribution: {w['contrib']['roughness']:.4f}",
        f"Semantic contribution: {w['contrib']['semantic']:.4f}",
        "",
        f"Active source nodes: {w['kg_trace']['active_sources']}",
        f"Property activation: {w['kg_trace']['property_activation']}",
        f"Terrain penalties: {w['kg_trace']['terrain_penalty']}",
        "",
        "Top candidates:"
    ]

    for r in result["ranked"][:top]:
        lines.append(
            f"#{r['rank']} | "
            f"Location=({r['x']}, {r['y']}) | "
            f"Cost={r['cost']:.4f} | "
            f"D={r['D_norm']:.3f} | "
            f"R={r['R_norm']:.3f} | "
            f"S={r['S']:.3f} | "
            f"Terrain={r['terrain']}"
        )

    return "\n".join(lines)
