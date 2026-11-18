"""Backend for the demo. The UI only calls these functions."""
import numpy as np
import cv2

from segment import predict
from depth_module_v2 import DepthAnalysis
from candidates import generate_candidates
from cost import rank_candidates
from semantic_brain import (load_active_nodes, SOURCES, PROPS, TERRAINS,
                            E1, E2, _SCALE as KG_SCALE)
from class_policy import allowed_classes

W, H = 800, 600
DIAG = float(np.hypot(W, H))
DEFAULTS = dict(min_safe_frac=0.995, stride=16, max_candidates=50,
                allow_conditional=False, erode_px=15,
                use_probs=True, prob_safe_min=0.8)


# ---------- perception (run once per image) ----------
def analyze_image(image_rgb):
    img = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_AREA)
    pred, probs = predict(img, clean=True)
    return dict(image=img, pred=pred, probs=probs, da=DepthAnalysis(img))


# ---------- helpers ----------
def footprint_px(length_cm, width_cm, margin_cm, gsd_cm_per_px):
    """Required landing footprint in pixels; margin is added on every side."""
    w = (length_cm + 2 * margin_cm) / gsd_cm_per_px
    h = (width_cm + 2 * margin_cm) / gsd_cm_per_px
    return max(8, int(round(w))), max(8, int(round(h)))


def normalize_weights(kd, kr, ks):
    t = kd + kr + ks
    return (0.4, 0.2, 0.4) if t <= 0 else (kd / t, kr / t, ks / t)


def safe_fraction(c, allow_conditional):
    return float(c["class_hist"][sorted(allowed_classes(allow_conditional))].sum())


# ---------- candidate search + ranking ----------
def find_candidates(analysis, target, box_px, settings):
    s = {**DEFAULTS, **settings}
    kw = dict(box_w=box_px[0], box_h=box_px[1], min_safe_frac=s["min_safe_frac"],
              stride=int(s["stride"]), max_depth_checks=max(300, 3 * s["max_candidates"]),
              top_k=int(s["max_candidates"]), nms_iou=0.0,
              allow_conditional=s["allow_conditional"], erode_px=int(s["erode_px"]))
    if s["use_probs"]:
        kw.update(probs=analysis["probs"], prob_safe_min=s["prob_safe_min"])
    return generate_candidates(analysis["pred"], target, analysis["da"], **kw)


def plan_mission(analysis, target, package, weights, box_px, settings):
    s = {**DEFAULTS, **settings}
    res = find_candidates(analysis, target, box_px, s)
    kd, kr, ks = normalize_weights(*weights)
    plan = dict(status=res["status"], n_rejected=res["n_rejected"], target=tuple(target),
                package=package, box_px=box_px, settings=s, weights=(kd, kr, ks),
                n_candidates=len(res["candidates"]), ranked=None, baseline=None, active=None)
    if res["status"] != "ok":
        return plan
    pred = analysis["pred"]
    active = load_active_nodes({"package": package})
    ranked = rank_candidates(res["candidates"], pred, active, k=(kd, kr, ks))
    kb = kd + kr                                    # terrain-only baseline: k_S = 0
    base_k = (kd / kb, kr / kb, 0.0) if kb > 0 else (1.0, 0.0, 0.0)
    baseline = rank_candidates(res["candidates"], pred, active, k=base_k)
    plan.update(ranked=ranked, baseline=baseline, active=active)
    return plan


# ---------- explanation (built from real metrics and KG traces) ----------
def explain_ranking(plan, steep_limit, n_alt=2):
    ranked = plan["ranked"]["ranked"]
    w = ranked[0]
    tr, pen = w["kg_trace"], w["kg_trace"]["terrain_penalty"]
    srcs = tr["active_sources"] or ["none"]
    sf = safe_fraction(w, plan["settings"]["allow_conditional"])
    why = [
        f"Safe class: dominant terrain is {w['terrain']}, and {sf * 100:.1f}% of the footprint lies on classes accepted as safe.",
        f"Geometry: the box lies fully inside the image, slope {w['slope']:.3f} (limit {steep_limit}), depth roughness {w['R']:.3f}.",
        f"Distance: {w['D'] * DIAG:.0f} px from the target (normalised {w['D_norm']:.2f}), contributing {w['contrib']['distance']:.3f} to the cost.",
        f"Semantic: active sources {srcs} activate {tr['property_activation'] or 'no properties'}; "
        f"the penalty for {w['terrain']} is {w['S']:.3f}, contributing {w['contrib']['semantic']:.3f}.",
        f"Total weighted cost {w['cost']:.3f} is the lowest of {len(ranked)} valid candidates.",
    ]
    versus = []
    for alt in ranked[1:1 + n_alt]:
        d = {k: alt["contrib"][k] - w["contrib"][k] for k in ("distance", "roughness", "semantic")}
        driver = max(d, key=d.get)
        t = (f"#{alt['rank']} at ({alt['x']},{alt['y']}) on {alt['terrain']} costs {alt['cost']:.3f} "
             f"({alt['cost'] - w['cost']:+.3f} vs winner; distance {d['distance']:+.3f}, "
             f"roughness {d['roughness']:+.3f}, semantic {d['semantic']:+.3f}). Largest driver: {driver}.")
        if driver == "semantic":
            t += (f" Under {', '.join(srcs)} the KG penalty is {pen[alt['terrain']]:.3f} for "
                  f"{alt['terrain']} vs {pen[w['terrain']]:.3f} for {w['terrain']}.")
        elif driver == "distance":
            t += f" It is {(alt['D'] - w['D']) * DIAG:.0f} px farther from the target."
        else:
            t += " Its depth roughness is higher."
        versus.append(t)
    return why, versus


# ---------- package comparison ----------
def compare_packages(analysis, target, box_px, settings, weights, scenarios):
    s = {**DEFAULTS, **settings}
    res = find_candidates(analysis, target, box_px, s)
    if res["status"] != "ok":
        return dict(status=res["status"], rows=[], baseline=None)
    cands, pred = res["candidates"], analysis["pred"]
    kd, kr, ks = normalize_weights(*weights)
    rows, ref = [], None
    for name, pkg in scenarios.items():
        active = load_active_nodes({"package": pkg})
        out = rank_candidates(cands, pred, active, k=(kd, kr, ks))
        w = out["winner"]
        pen = w["kg_trace"]["terrain_penalty"]
        if ref is None:
            ref, changed, why = w, "reference", "First scenario, used as the reference."
        elif (w["x"], w["y"]) == (ref["x"], ref["y"]):
            changed = "no"
            why = f"Same site: {w['terrain']} stays cheapest (penalty {pen[w['terrain']]:.3f})."
        else:
            changed = "yes"
            why = (f"Reference terrain {ref['terrain']} now has penalty {pen[ref['terrain']]:.3f} vs "
                   f"{pen[w['terrain']]:.3f} for {w['terrain']}; weighted cost {w['cost']:.3f}.")
        rows.append(dict(Scenario=name, Active=", ".join(w["kg_trace"]["active_sources"]) or "none",
                         Pavement=pen["Pavement"], Grass=pen["Grass"], Dirt=pen["Dirt"],
                         Winner=f"({w['x']}, {w['y']})", Terrain=w["terrain"],
                         Cost=round(w["cost"], 4), Changed=changed, Why=why,
                         _ranked=out["ranked"]))
    kb = kd + kr
    base_k = (kd / kb, kr / kb, 0.0) if kb > 0 else (1.0, 0.0, 0.0)
    base = rank_candidates(cands, pred, load_active_nodes({"package": {}}), k=base_k)
    return dict(status="ok", rows=rows, baseline=base)


# ---------- knowledge-graph views ----------
def kg_graph_dot(active, penalties):
    srcs = [s for s, v in zip(SOURCES, active) if v > 0]
    L = ["digraph G {", "rankdir=LR;", 'node [shape=box, style=filled, fontname="Helvetica"];']
    for s in srcs:
        L.append(f'"{s}" [fillcolor="#f4b6b6"];')
    props = set()
    for (s, p), w in E1.items():
        if s in srcs:
            props.add(p)
            L.append(f'"{s}" -> "{p}" [label="{w:.2f}", penwidth={1 + 3 * w:.1f}];')
    for p in sorted(props):
        L.append(f'"{p}" [fillcolor="#f6e3a1"];')
    for (p, t), w in E2.items():
        if p in props and w > 0:
            L.append(f'"{p}" -> "{t}" [label="{w:.2f}", penwidth={1 + 3 * w:.1f}];')
    for t in TERRAINS:
        L.append(f'"{t}" [label="{t}\\npenalty {penalties[t]:.3f}", fillcolor="#b6e3b6"];')
    L.append("}")
    return "\n".join(L)


def kg_paths(active, terrain):
    rows = []
    for s, v in zip(SOURCES, active):
        if v <= 0:
            continue
        for p in PROPS:
            w1, w2 = E1.get((s, p), 0.0), E2.get((p, terrain), 0.0)
            if w1 and w2:
                rows.append({"Path": f"{s} → {p} → {terrain}", "Edge 1": w1, "Edge 2": w2,
                             "Contribution": round(w1 * w2 / KG_SCALE, 3)})
    return sorted(rows, key=lambda r: -r["Contribution"])