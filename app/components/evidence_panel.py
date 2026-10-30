"""app/components/evidence_panel.py — Retrieved evidence display component."""
import streamlit as st
from src.retrieval.schemas import RetrievalResult


def render_evidence(retrieval: RetrievalResult):
    """Render the retrieved cultural evidence with source citations."""
    st.markdown("---")
    st.markdown('<div class="section-header">📚 Retrieved Cultural Evidence</div>', unsafe_allow_html=True)

    if not retrieval.evidence_found:
        st.warning(f"⚠️ No cultural evidence retrieved. {retrieval.retrieval_message}")
        return

    st.caption(f"Query: `{retrieval.query_used[:100]}{'...' if len(retrieval.query_used) > 100 else ''}`")
    st.caption(f"Retrieved {len(retrieval.retrieved_records)} records from index of {retrieval.num_records_in_index}")

    for e in retrieval.retrieved_records:
        confidence_colour = {
            "verified": "🟢", "plausible": "🟡", "synthetic_demo": "🔴"
        }.get(e.confidence, "⚪")

        with st.expander(
            f"[#{e.retrieval_rank}] {e.topic} — Score: {e.relevance_score:.2f} {confidence_colour}",
            expanded=(e.retrieval_rank == 1),
        ):
            if e.is_synthetic:
                st.error("⚠️ SYNTHETIC RECORD — Not a verified cultural fact. For demonstration only.")

            st.markdown(f"**Region:** {e.cultural_region}")
            st.markdown(f"**Evidence:** {e.evidence_text}")
            st.markdown(f"**Confidence:** `{e.confidence}`")
            st.caption(f"⚠️ Limitations: {e.known_limitations}")

            if e.source_url != "NOT_AVAILABLE":
                st.markdown(f"📎 **Source:** [{e.source_title}]({e.source_url})")
            else:
                st.caption(f"Source: {e.source_title} (no URL available)")

            if e.tags:
                st.markdown(" ".join([f"`{t}`" for t in e.tags if t]))
