"""app/components/consistency_warnings.py — Consistency warnings display component."""
import streamlit as st
from src.verification.schemas import ConsistencyReport, WarningSeverity


def render_warnings(report: ConsistencyReport):
    """Render cultural consistency warnings with severity indicators."""
    st.markdown("---")
    st.markdown('<div class="section-header">🛡️ Cultural Consistency Check</div>', unsafe_allow_html=True)

    status_colours = {"pass": "success", "caution": "warning", "fail": "error"}
    status_icons = {"pass": "✅", "caution": "⚠️", "fail": "❌"}

    icon = status_icons.get(report.overall_status, "ℹ️")
    msg_fn = getattr(st, status_colours.get(report.overall_status, "info"))
    msg_fn(f"{icon} {report.summary}")

    if report.warnings:
        for w in report.warnings:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w.severity.value, "⚪")
            with st.expander(f"{severity_icon} [{w.category.upper()}] {w.message[:80]}...", expanded=(w.severity == WarningSeverity.HIGH)):
                st.markdown(f"**Category:** `{w.category}`")
                st.markdown(f"**Severity:** `{w.severity.value}`")
                st.markdown(f"**Issue:** {w.message}")
                if w.suggested_revision:
                    st.info(f"💡 **Suggested revision:** {w.suggested_revision}")
