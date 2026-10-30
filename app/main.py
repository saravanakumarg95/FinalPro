"""
app/main.py
────────────────────────────────────────────────────────────────
Streamlit application entry point.

Run:
    streamlit run app/main.py

The app provides:
  - Image upload and validation
  - Tamil cultural story generation (proposed method)
  - Retrieved cultural evidence display with sources
  - Reasoning trace inspection
  - Cultural consistency warnings
  - Baseline comparison (when API available)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# Page config MUST be the very first Streamlit call
st.set_page_config(
    page_title="Tamil Cultural Story Generator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.image_upload import render_image_upload
from app.components.story_display import render_story
from app.components.evidence_panel import render_evidence
from app.components.reasoning_trace import render_reasoning_trace
from app.components.consistency_warnings import render_warnings
from src.config.settings import AppConfig, get_settings
from src.utils.logger import setup_logger

# ── Initialise ─────────────────────────────────────────────────────────────────
config = get_settings()
setup_logger(config.log_level.value)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

:root {
    --primary: #C84B31;      /* Tamil red-orange */
    --secondary: #2C3E50;
    --accent: #F39C12;       /* Gold */
    --bg-card: #1E1E2E;
    --text-light: #ECF0F1;
    --success: #27AE60;
    --warning: #E67E22;
}

body { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #C84B31 0%, #8B2500 50%, #2C3E50 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(200, 75, 49, 0.3);
}

.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: #FDEBD0;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.main-header p {
    color: #FAD7A0;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.card {
    background: linear-gradient(145deg, #1E1E2E, #252535);
    border: 1px solid rgba(200, 75, 49, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F39C12;
    border-bottom: 2px solid rgba(243, 156, 18, 0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.evidence-badge {
    display: inline-block;
    background: rgba(200, 75, 49, 0.2);
    color: #E74C3C;
    border: 1px solid rgba(200, 75, 49, 0.4);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}

.verified-badge {
    background: rgba(39, 174, 96, 0.2);
    color: #2ECC71;
    border-color: rgba(39, 174, 96, 0.4);
}

.story-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #ECF0F1;
    background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
    padding: 2rem;
    border-radius: 12px;
    border-left: 4px solid #C84B31;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.3);
}

.mode-badge {
    background: rgba(200, 75, 49, 0.15);
    color: #E74C3C;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.live-badge {
    background: rgba(39, 174, 96, 0.15);
    color: #2ECC71;
}

.stButton > button {
    background: linear-gradient(135deg, #C84B31, #8B2500) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(200, 75, 49, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(200, 75, 49, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ Tamil Cultural Story Generator</h1>
    <p>Evidence-Grounded Cultural Reasoning from Images · Final Year Research Project</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    mode_label = "🟢 Live Mode" if config.is_live_mode else "🟡 Mock Mode"
    st.markdown(f"**Pipeline Mode:** {mode_label}")

    if config.is_mock_mode:
        st.info(
            "Running in **mock mode** — results use deterministic responses. "
            "Add `GEMINI_API_KEY` to `.env` and set `APP_MODE=live` for real generation."
        )

    top_k = st.slider("🔍 Evidence records to retrieve (top-k)", 1, 10, config.retrieval_top_k)
    target_words = st.slider("📝 Story length (words)", 100, 800, config.story_target_words, step=50)
    reading_level = st.selectbox(
        "📖 Reading level",
        ["general", "simple", "academic"],
        index=0,
    )
    show_trace = st.checkbox("🔬 Show reasoning trace", value=False)
    show_baseline = st.checkbox("📊 Show baseline comparison", value=False)

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown(
        "This system generates **Tamil cultural stories** from images using:\n"
        "- **Visual understanding** (Gemini Flash VLM)\n"
        "- **Cultural KB retrieval** (ChromaDB + MiniLM)\n"
        "- **Evidence-grounded reasoning** (7-step pipeline)\n"
        "- **Cultural consistency checks** (rule-based)\n\n"
        "**Research goal:** Compare evidence-grounded generation vs. direct and fixed-example baselines."
    )
    st.markdown("---")
    st.caption("⚠️ Synthetic KB records are clearly labelled and never presented as verified facts.")

# ── Main content ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="section-header">📷 Upload Image</div>', unsafe_allow_html=True)
    uploaded_image, image_bytes, pil_image = render_image_upload()

    if pil_image:
        st.markdown('<div class="section-header">🖼️ Image Preview</div>', unsafe_allow_html=True)
        st.image(pil_image, width="stretch")

        st.markdown("---")
        generate_clicked = st.button(
            "🎭 Generate Tamil Cultural Story",
            use_container_width=True,
            key="generate_btn",
        )
    else:
        generate_clicked = False

with col2:
    if pil_image and generate_clicked:
        # ── Run pipeline ───────────────────────────────────────────────────
        with st.spinner("🔍 Analysing image..."):
            from app.components.pipeline_runner import run_pipeline
            try:
                result = run_pipeline(
                    pil_image=pil_image,
                    config=config,
                    top_k=top_k,
                    target_words=target_words,
                    reading_level=reading_level,
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                st.exception(e)
                result = None

        if result:
            # Visual observations
            with st.expander("👁️ Visual Observations", expanded=True):
                obs = result["observation"]
                st.markdown(f"**Setting:** {obs.setting}")
                if obs.activities:
                    st.markdown("**Activities:** " + " · ".join(obs.activities))
                if obs.cultural_clues:
                    st.markdown("**Cultural clues observed:**")
                    for clue in obs.cultural_clues:
                        st.markdown(f"  - {clue}")
                if obs.uncertainty_notes:
                    st.warning("**Uncertainties:** " + " | ".join(obs.uncertainty_notes))

            # Retrieved evidence
            render_evidence(result["retrieval"])

            # Consistency warnings
            render_warnings(result["consistency"])

            # Story
            render_story(result["story"])

            # Reasoning trace (optional)
            if show_trace:
                render_reasoning_trace(result["trace"])

            # Baseline comparison (optional)
            if show_baseline and result.get("baselines"):
                st.markdown("---")
                st.markdown('<div class="section-header">📊 Baseline Comparison</div>', unsafe_allow_html=True)
                tab_a, tab_b, tab_proposed = st.tabs(["Baseline A (Direct)", "Baseline B (Fixed Examples)", "Proposed (Evidence-Grounded)"])
                with tab_a:
                    st.write(result["baselines"]["baseline_a"].story_text if result["baselines"].get("baseline_a") else "Not available")
                with tab_b:
                    st.write(result["baselines"]["baseline_b"].story_text if result["baselines"].get("baseline_b") else "Not available")
                with tab_proposed:
                    st.write(result["story"].story_text)

    elif not pil_image:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; opacity: 0.6;">
            <div style="font-size: 4rem;">🏛️</div>
            <h3 style="color: #F39C12;">Upload an image to begin</h3>
            <p>Upload any image containing Tamil cultural elements — festivals, temples, 
            clothing, food, music, or daily life — and the system will generate a 
            culturally grounded story using retrieved evidence.</p>
        </div>
        """, unsafe_allow_html=True)
