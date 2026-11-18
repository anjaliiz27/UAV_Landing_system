import json
import colorsys
import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import pipeline as P
from class_policy import CLASS_NAMES, allowed_classes

st.set_page_config(page_title="UAV Landing-Site Optimiser", layout="wide")

# ================= styling =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"], button, input, label {
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
h1 {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.8rem !important;
}
.stButton > button[kind="primary"] {
    background-color: #287A5A !important;
    border-color: #287A5A !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: 6px !important;
    width: 100% !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    padding-bottom: 5px !important;
}
.stTabs [data-baseweb="tab"] {
    flex: 0 0 auto !important;
    padding: 8px 13px !important;
    min-height: 40px !important;
    height: 40px !important;
    background: #F3F7F4 !important;
    border: 1px solid #D5E2D9 !important;
    border-radius: 9px !important;
    white-space: nowrap !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"] p {
    font-family: 'Manrope', sans-serif !important;
    font-size: 13px !important;
    font-weight: 650 !important;
    letter-spacing: 0 !important;
    color: #34483C !important;
    margin: 0 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #E7F0EA !important;
    border-color: #9AB9A5 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #287A5A !important;
    border-color: #287A5A !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}
[data-testid="stAppDeployButton"] {
    display: none !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #DCEBE1 !important;
    border: 1px solid #B7D2C0 !important;
    border-radius: 7px !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #285C43 !important;
    font-weight: 600 !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    fill: #285C43 !important;
    color: #285C43 !important;
}
.stMultiSelect [data-baseweb="tag"]:hover {
    background-color: #C8DFD0 !important;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
W, H = P.W, P.H

PALETTE = np.array([
    [int(c * 255) for c in colorsys.hsv_to_rgb(i / 24, 0.65, 0.95)]
    for i in range(24)
], np.uint8)
PALETTE[0] = (40, 40, 40)

PRESETS = {
    "A · Fragile + Valuable": {"fragile": True, "valuable": True},
    "B · Heavy": {"heavy": True},
    "C · Biohazard": {"biohazard": True},
    "D · All four": {
        "fragile": True, "valuable": True, "biohazard": True, "heavy": True
    },
    "E · No special properties": {},
}

@st.cache_resource(show_spinner=False)
def get_analysis(data: bytes):
    """Run the perception models once for these exact image bytes."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The selected file could not be decoded as an image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return P.analyze_image(rgb)


def draw(image, target, ranked=None, n_alt=4):
    img = image.copy()
    if ranked:
        for i, c in reversed(list(enumerate(ranked[:n_alt + 1]))):
            pts = np.round(c["corners"]).astype(np.int32)
            cv2.polylines(
                img, [pts], True,
                (0, 255, 0) if i == 0 else (255, 220, 0),
                3 if i == 0 else 2
            )
            cv2.putText(
                img, str(i + 1), (int(c["x"]) - 6, int(c["y"]) + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    cv2.drawMarker(
        img, (int(target[0]), int(target[1])), (255, 0, 0),
        cv2.MARKER_STAR, 26, 3
    )
    return img


# ================= sidebar: image selection =================
st.sidebar.title("Mission setup")
up = st.sidebar.file_uploader(
    "Aerial image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)
samples = sorted((ROOT / "test_images").glob("*.jpg"))

if up is not None:
    data = up.getvalue()
elif samples:
    choice = st.sidebar.selectbox(
        "...or a sample image", [s.name for s in samples]
    )
    data = (ROOT / "test_images" / choice).read_bytes()
else:
    st.title("UAV Landing-Site Optimiser")
    st.info("Upload an aerial image to start, or add JPG samples to the test_images folder.")
    st.stop()

# Decode and resize for display immediately. This does NOT run either AI model.
bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
    st.error("The selected file is not a valid JPG/PNG image.")
    st.stop()
preview_image = cv2.cvtColor(
    cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA),
    cv2.COLOR_BGR2RGB
)

# If the selected image changes, discard results belonging to the old image.
image_key = hashlib.sha256(data).hexdigest()
S = st.session_state
if S.get("image_key") != image_key:
    S.image_key = image_key
    S.pop("analysis", None)
    S.pop("plan", None)
    S.pop("cmp", None)
    S.pop("last_click", None)
    S.tx, S.ty = W // 2, H // 2

# ================= sidebar: mission settings =================
st.sidebar.subheader("Package")
package = {
    "fragile": st.sidebar.checkbox("Fragile"),
    "heavy": st.sidebar.checkbox("Heavy"),
    "valuable": st.sidebar.checkbox("Valuable"),
    "biohazard": st.sidebar.checkbox("Biohazard")
}

length = st.sidebar.number_input("Length (cm)", 10, 500, 100)
width = st.sidebar.number_input("Width (cm)", 10, 500, 60)
st.sidebar.number_input(
    "Height (cm)", 1, 500, 30, help="Metadata only for now"
)
margin = st.sidebar.number_input("Safety margin (cm)", 0, 200, 10)
gsd = st.sidebar.number_input(
    "Image scale (cm per pixel)", 0.1, 50.0, 2.0, 0.1,
    help="Assumption: not measured. Real value depends on altitude and camera."
)
box_px = P.footprint_px(length, width, margin, gsd)
st.sidebar.caption(f"Required footprint: {box_px[0]} × {box_px[1]} px")

st.sidebar.subheader("Cost weights (sum = 1)")
kd = st.sidebar.slider("k_D (distance)", 0.0, 1.0, 0.4, 0.05)
kr = st.sidebar.slider("k_R (roughness)", 0.0, 1.0, 0.2, 0.05)
ks = round(1.0 - kd - kr, 2)
if ks < 0:
    st.sidebar.error("k_D + k_R must not exceed 1.")
    ks = 0.0
st.sidebar.write(f"k_S (semantic) = **{ks:.2f}**")

with st.sidebar.expander("Advanced settings"):
    settings = dict(
        min_safe_frac=st.slider(
            "Minimum safe-pixel fraction", 0.80, 1.0, 0.995, 0.005
        ),
        stride=st.slider("Search stride (px)", 4, 32, 16),
        max_candidates=st.slider("Maximum candidates", 5, 100, 50),
        allow_conditional=st.checkbox("Allow gravel / vegetation", False),
        erode_px=st.slider("Safety margin around hazards (px)", 1, 31, 15),
        use_probs=st.checkbox("Require model confidence", True),
        prob_safe_min=st.slider("Minimum P(safe)", 0.5, 0.99, 0.8, 0.01),
    )

# Target state
S.setdefault("tx", W // 2)
S.setdefault("ty", H // 2)

def reset_target():
    S.tx, S.ty = W // 2, H // 2
    S.pop("plan", None)

# Analysis is available only after the user explicitly requests it.
A = S.get("analysis") if S.get("image_key") == image_key else None
plan = S.get("plan") if A is not None else None
ready = bool(plan and plan.get("status") == "ok")

# ================= main page =================
st.title("UAV Landing-Site Optimiser")
tabs = st.tabs([
    "Mission", "Perception", "Result", "Why this site",
    "Knowledge graph", "Package comparison"
])

# ---- tab 1: mission / target selection ----
with tabs[0]:
    left, right = st.columns([3, 1])
    with left:
        st.caption("Click the image to set the target (it can be any pixel, even an unsafe one).")
        click = streamlit_image_coordinates(
            draw(preview_image, (S.tx, S.ty)),
            width=W,
            key=f"pick_{image_key[:10]}"
        )
        if click and (click["x"], click["y"]) != S.get("last_click"):
            S.last_click = (click["x"], click["y"])
            S.tx = int(np.clip(click["x"] * W / click["width"], 0, W - 1))
            S.ty = int(np.clip(click["y"] * H / click["height"], 0, H - 1))
            S.pop("plan", None)
            st.rerun()

    with right:
        st.number_input("Target x", -100, W + 100, key="tx")
        st.number_input("Target y", -100, H + 100, key="ty")
        st.caption("Values outside the image give the 'invalid target' result.")
        st.button("Reset target", on_click=reset_target)
        run = st.button(
            "Find landing site", type="primary", use_container_width=True
        )

    if run:
        try:
            with st.spinner("Loading U-Net and MiDaS, then searching for the best landing site..."):
                if A is None:
                    A = get_analysis(data)
                    S.analysis = A
                S.plan = P.plan_mission(
                    A, (S.tx, S.ty), package, (kd, kr, ks),
                    box_px, settings
                )
                plan = S.plan
                ready = bool(plan and plan.get("status") == "ok")
            st.success("Analysis complete.")
        except Exception as e:
            st.error("Image analysis or landing-site search failed.")
            st.exception(e)

    if A is None:
        st.caption("The image and mission controls are ready. Model inference starts only when you click **Find landing site**.")
    elif not plan:
        st.caption("Perception is loaded. Click **Find landing site** to rank candidate landing sites.")

# ---- tab 2: perception ----
with tabs[1]:
    if A is None:
        st.info("Run the mission using **Find landing site** to generate U-Net segmentation and MiDaS depth outputs.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.image(PALETTE[A["pred"]], caption="Semantic segmentation (U-Net)")
        d = A["da"].depth
        c2.image(
            cv2.cvtColor(
                cv2.applyColorMap((255 * d).astype(np.uint8), cv2.COLORMAP_INFERNO),
                cv2.COLOR_BGR2RGB
            ),
            caption="Relative depth (MiDaS)"
        )
        ok = np.isin(A["pred"], sorted(allowed_classes(settings["allow_conditional"])))
        ov = A["image"].copy()
        ov[ok] = (0.6 * ov[ok] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
        ov[~ok] = (0.6 * ov[~ok] + 0.4 * np.array([220, 0, 0])).astype(np.uint8)
        c3.image(ov, caption="Safe (green) and rejected (red) terrain classes")
        st.subheader("Legend (classes present)")
        cols = st.columns(6)
        for i, cid in enumerate(np.unique(A["pred"])):
            cols[i % 6].markdown(
                f"<span style='background:rgb{tuple(int(v) for v in PALETTE[cid])};"
                f"padding:0 10px'>&nbsp;</span> {cid} {CLASS_NAMES[int(cid)]}",
                unsafe_allow_html=True
            )

# ---- tab 3: result ----
with tabs[2]:
    if not plan:
        st.info("Set a target and press 'Find landing site' in the Mission tab.")
    elif plan["status"] == "invalid_target":
        st.error(f"Invalid target {plan['target']}: it lies outside the image.")
    elif plan["status"] == "no_valid_candidate":
        st.warning(
            "No valid landing site found. Try a smaller package, a lower safe-pixel "
            "fraction, or allow gravel / vegetation in the advanced settings."
        )
    else:
        ranked = plan["ranked"]["ranked"]
        w = ranked[0]
        base_w = plan["baseline"]["winner"]
        m = st.columns(4)
        m[0].metric("Selected terrain", w["terrain"])
        m[1].metric("Landing location", f"({w['x']}, {w['y']})")
        m[2].metric("Total cost", f"{w['cost']:.4f}")
        m[3].metric("Target", str(plan["target"]))
        annotated = draw(A["image"], plan["target"], ranked)
        a, b = st.columns([3, 2])
        a.image(annotated, caption="Green: winner · Yellow: alternatives · Red star: target")
        k_d, k_r, k_s = plan["weights"]
        b.markdown(f"**C = {k_d:.2f}·D + {k_r:.2f}·R + {k_s:.2f}·S**")
        b.dataframe([
            {
                "Rank": r["rank"], "Location": f"({r['x']},{r['y']})",
                "Terrain": r["terrain"], "Dist px": round(r["D"] * P.DIAG),
                "D": round(r["D_norm"], 3), "R": round(r["R_norm"], 3),
                "S": round(r["S"], 3), "Slope": round(r["slope"], 3),
                "Cost": round(r["cost"], 4)
            } for r in ranked[:10]
        ], hide_index=True)
        with st.expander("Safety evidence", expanded=True):
            lim = A["da"].steep_threshold
            st.write(f"- Safe-class coverage of winner: {P.safe_fraction(w, plan['settings']['allow_conditional']) * 100:.1f}%")
            st.write(f"- Slope {w['slope']:.3f} (limit {lim}) · depth roughness {w['R']:.3f}")
            st.write(f"- Valid candidates: {plan['n_candidates']} · rejected by depth check: {plan['n_rejected']}")
            st.write(
                f"- Terrain-only baseline (k_S = 0) picks ({base_w['x']}, {base_w['y']}) "
                f"on {base_w['terrain']}; payload-aware picks ({w['x']}, {w['y']}) "
                f"on {w['terrain']}."
            )
        summary = dict(
            target=plan["target"], package=plan["package"], weights=plan["weights"],
            footprint_px=plan["box_px"], settings=plan["settings"],
            winner={k: w[k] for k in ("x", "y", "angle", "terrain", "cost", "D_norm", "R_norm", "S")},
            ranking=[
                {k: r[k] for k in ("rank", "x", "y", "terrain", "cost")}
                for r in ranked[:10]
            ],
            terrain_penalties=w["kg_trace"]["terrain_penalty"]
        )
        d1, d2 = st.columns(2)
        d1.download_button("Download summary (JSON)", json.dumps(summary, indent=2, default=float), "landing_summary.json")
        d2.download_button(
            "Download result image (PNG)",
            cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))[1].tobytes(),
            "landing_result.png", "image/png"
        )

# ---- tab 4: explanation ----
with tabs[3]:
    if not ready:
        st.info("Run the search first.")
    else:
        why, versus = P.explain_ranking(plan, A["da"].steep_threshold)
        st.subheader("Why was this landing site selected?")
        for t in why:
            st.write("- " + t)
        st.subheader("Why not the alternatives?")
        for t in versus:
            st.write("- " + t)
        ranked = plan["ranked"]["ranked"]
        st.subheader("Cost contributions")
        st.dataframe([
            {
                "Rank": r["rank"], "Terrain": r["terrain"],
                "distance": round(r["contrib"]["distance"], 4),
                "roughness": round(r["contrib"]["roughness"], 4),
                "semantic": round(r["contrib"]["semantic"], 4),
                "total": round(r["cost"], 4)
            } for r in ranked[:3]
        ], hide_index=True)

# ---- tab 5: knowledge graph ----
with tabs[4]:
    if not ready:
        st.info("Run the search first.")
    else:
        w = plan["ranked"]["ranked"][0]
        tr = w["kg_trace"]
        st.write("**Active Layer 1 (package) nodes:**", ", ".join(tr["active_sources"]) or "none")
        st.write("**Activated Layer 2 (property) nodes:**", tr["property_activation"] or "none")
        st.graphviz_chart(P.kg_graph_dot(plan["active"], tr["terrain_penalty"]))
        st.write("**Layer 3 terrain penalties:**", tr["terrain_penalty"])
        st.subheader(f"Edge paths behind the {w['terrain']} penalty (winner's terrain)")
        st.dataframe(P.kg_paths(plan["active"], w["terrain"]), hide_index=True)

# ---- tab 6: package comparison ----
with tabs[5]:
    st.caption("Same image, target and weights; only the package changes.")
    chosen = st.multiselect("Scenarios", list(PRESETS), default=list(PRESETS)[:4])
    if A is None:
        st.info("Run the mission first to load the perception models.")
    else:
        if st.button("Run comparison") and chosen:
            try:
                with st.spinner("Comparing package scenarios..."):
                    S.cmp = P.compare_packages(
                        A, (S.tx, S.ty), box_px, settings, (kd, kr, ks),
                        {n: PRESETS[n] for n in chosen}
                    )
            except Exception as e:
                st.error("Package comparison failed.")
                st.exception(e)
        cmp = S.get("cmp")
        if cmp:
            if cmp["status"] != "ok":
                st.warning(f"Comparison not possible: {cmp['status']}.")
            else:
                st.dataframe([
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in cmp["rows"]
                ], hide_index=True)
                bw = cmp["baseline"]["winner"]
                distinct = len({r["Winner"] for r in cmp["rows"]})
                st.info(
                    f"Terrain-only baseline (no knowledge graph) always picks "
                    f"({bw['x']}, {bw['y']}) on {bw['terrain']}. Payload-aware "
                    f"reasoning picked {distinct} distinct site(s) across "
                    f"{len(cmp['rows'])} scenarios."
                )
                cols = st.columns(len(cmp["rows"]))
                for col, r in zip(cols, cmp["rows"]):
                    col.image(
                        draw(A["image"], (S.tx, S.ty), r["_ranked"], n_alt=2),
                        caption=r["Scenario"]
                    )
