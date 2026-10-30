"""app/components/story_display.py — Generated story display component."""
import streamlit as st
from src.generation.schemas import GeneratedStory


def render_story(story: GeneratedStory):
    """Render the generated story with word count and source info."""
    st.markdown("---")
    st.markdown('<div class="section-header">📖 Generated Tamil Cultural Story</div>', unsafe_allow_html=True)

    method_labels = {
        "proposed": "✨ Proposed (Evidence-Grounded)",
        "baseline_a": "Baseline A (Direct Generation)",
        "baseline_b": "Baseline B (Fixed Examples)",
        "mock": "Mock Output",
    }
    method = method_labels.get(story.generation_method, story.generation_method)
    st.caption(f"Method: {method} · {story.word_count} words · Model: {story.model_used or 'mock'}")

    st.markdown(
        f'<div class="story-text">{story.story_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    if story.cultural_elements_included:
        st.markdown("**Cultural elements included:**")
        cols = st.columns(min(4, len(story.cultural_elements_included)))
        for i, elem in enumerate(story.cultural_elements_included):
            cols[i % 4].markdown(f"🏷️ `{elem}`")
