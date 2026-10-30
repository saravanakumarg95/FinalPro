"""app/components/reasoning_trace.py — Reasoning trace inspector component."""
import streamlit as st
from src.reasoning.schemas import ReasoningTrace


def render_reasoning_trace(trace: ReasoningTrace):
    """Render the full reasoning trace for research inspection."""
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Reasoning Trace (Research Inspector)</div>', unsafe_allow_html=True)
    st.caption("This section shows the system's internal reasoning steps — the core research contribution.")

    with st.expander("Step 1 — Observation Summary", expanded=False):
        st.text(trace.observation_summary)
        if trace.key_cultural_clues:
            st.markdown("**Key cultural clues:** " + " · ".join(trace.key_cultural_clues))

    with st.expander("Step 2 — Retrieval", expanded=False):
        st.markdown(f"**Query used:** `{trace.retrieval_query_used}`")
        st.markdown(f"**Records retrieved:** {trace.total_evidence_retrieved}")
        st.markdown(f"**Record IDs:** {', '.join(trace.retrieved_evidence_ids)}")

    with st.expander("Step 3 — Evidence-Observation Alignment", expanded=False):
        for a in trace.evidence_alignments:
            use_icon = "✅" if a.use_in_story else "❌"
            st.markdown(
                f"{use_icon} **{a.evidence_topic}** — alignment: `{a.alignment_score:.3f}` · "
                f"use in story: `{a.use_in_story}`"
            )
            if a.aligned_observations:
                st.caption("Aligned terms: " + ", ".join(a.aligned_observations))

    with st.expander("Step 4 — Uncertainty & Contradictions", expanded=False):
        st.markdown(f"**Evidence coverage:** `{trace.evidence_coverage}`")
        for u in trace.uncertainties_identified:
            st.markdown(f"⚠️ {u}")
        for c in trace.contradictions_found:
            st.error(c)

    with st.expander("Step 5 — Story Outline", expanded=False):
        if trace.story_outline:
            o = trace.story_outline
            st.markdown(f"**Setting:** {o.setting}")
            st.markdown(f"**Character:** {o.main_character}")
            st.markdown(f"**Cultural elements to use:** {', '.join(o.cultural_elements_to_use)}")
            st.markdown(f"**Narrative arc:** {o.narrative_arc}")
            st.markdown(f"**Avoid:** {', '.join(o.cultural_claims_to_avoid)}")

    with st.expander("Step 6 — Pre-Generation Warnings", expanded=False):
        if trace.pre_generation_warnings:
            for w in trace.pre_generation_warnings:
                st.warning(w)
        else:
            st.success("No pre-generation warnings.")

    st.caption(f"Evidence used: {trace.total_evidence_used}/{trace.total_evidence_retrieved} · Pipeline v{trace.pipeline_version}")
