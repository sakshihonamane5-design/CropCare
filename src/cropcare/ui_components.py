"""CropCare ΓÇö Modular UI Component Library.

Provides reusable presentation components for Streamlit dashboards:
- Metric badges and status pills
- Step progress indicators
- Diagnostic cards with risk highlights
- Audio playback widgets
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st


def render_badge(text: str, bg_color: str = "#EBF8D3", text_color: str = "#123D2A") -> str:
    """Render an inline HTML pill badge."""
    return f"""
    <span style="display: inline-block; background-color: {bg_color}; color: {text_color};
    padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;
    border: 1px solid {text_color}33;">
        {text}
    </span>
    """


def render_alignment_gauge(score: int, badge: str, color: str = "#15803D"):
    """Render a visual Biometeorological Concurrence Gauge card."""
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #D6E0D9; border-radius: 16px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Alignment Index</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 800; color: {color};">{score}<span style="font-size: 1.2rem; color: #526E60;">/100</span></div>
            <div style="display: inline-block; background: {color}1A; color: {color}; font-weight: 700; font-size: 0.85rem; padding: 0.35rem 0.85rem; border-radius: 20px; border: 1px solid {color}33; margin-top: 0.5rem;">
                {badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_spray_banner(badge: str, advice: str):
    """Render a high-visibility spray advisory callout banner."""
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border-left: 5px solid #1B543A; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Spray Application Guidance</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: #123D2A; margin: 0.3rem 0;">{badge}</div>
            <div style="font-size: 0.92rem; color: #334D3F;">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
