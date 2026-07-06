# =============================================================================
#  Shared utilities: MIST-inspired theme, header, smooth-transition CSS
#  Imported by Home.py and every page in pages/
# =============================================================================
import os
import streamlit as st

LOGO_FILE = os.path.join(os.path.dirname(__file__), "mist_logo.jpg")

# Colors distilled from the MIST crest, muted for a professional / sober feel
COLORS = {
    "navy":        "#1B3A5C",   # deep blue gear
    "green":       "#1F5C43",   # muted institutional green
    "gold":        "#B8952E",   # muted gold/yellow ribbon
    "sky":         "#3E7CA6",   # muted sky blue
    "crimson":     "#8C2F2F",   # muted red
    "bg":          "#F7F8F7",
    "card_bg":     "#FFFFFF",
    "text_dark":   "#1C2B2A",
    "text_muted":  "#5A6663",
}


def inject_base_css():
    st.markdown(f"""
    <style>
        /* ---------- smooth fade-in for every page ---------- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .block-container {{
            animation: fadeInUp 0.55s ease-out;
        }}

        /* ---------- overall palette ---------- */
        .stApp {{
            background-color: {COLORS['bg']};
        }}

        h1, h2, h3 {{
            color: {COLORS['navy']};
            font-family: "Georgia", "Times New Roman", serif;
        }}

        p, li, span, label {{
            color: {COLORS['text_dark']};
        }}

        /* ---------- buttons: tasteful hover transitions ---------- */
        div[data-testid="stButton"] > button {{
            border-radius: 10px;
            border: 1px solid {COLORS['navy']}22;
            background-color: {COLORS['card_bg']};
            color: {COLORS['navy']};
            font-weight: 600;
            transition: all 0.25s ease-in-out;
            box-shadow: 0 1px 3px rgba(27,58,92,0.08);
        }}
        div[data-testid="stButton"] > button:hover {{
            background-color: {COLORS['navy']};
            color: white;
            border-color: {COLORS['navy']};
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(27,58,92,0.25);
        }}
        div[data-testid="stButton"] > button:active {{
            transform: translateY(0px);
        }}

        /* primary CTA buttons get the gold treatment */
        div[data-testid="stButton"] > button[kind="primary"] {{
            background-color: {COLORS['green']};
            color: white;
            border: none;
        }}
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            background-color: {COLORS['gold']};
            color: {COLORS['text_dark']};
        }}

        /* ---------- custom category cards ---------- */
        .category-card {{
            background-color: {COLORS['card_bg']};
            border-radius: 14px;
            padding: 1.6rem 1.4rem;
            border: 1px solid #00000010;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease-in-out;
            height: 100%;
        }}
        .category-card:hover {{
            box-shadow: 0 10px 24px rgba(27,58,92,0.15);
            transform: translateY(-4px);
        }}
        .category-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }}
        .category-desc {{
            color: {COLORS['text_muted']};
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }}
        .pill {{
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.15rem 0.25rem 0.15rem 0;
        }}
        .pill-mech {{ background-color: {COLORS['navy']}15; color: {COLORS['navy']}; }}
        .pill-dur  {{ background-color: {COLORS['sky']}18;  color: {COLORS['sky']}; }}

        hr {{
            border-top: 1px solid #00000012;
        }}

        /* ---------- back-to-home link ---------- */
        .back-link {{
            font-size: 0.9rem;
            color: {COLORS['text_muted']};
        }}
    </style>
    """, unsafe_allow_html=True)


def show_header(subtitle=None):
    """Small shared header with logo, used at the top of sub-pages."""
    col_logo, col_text = st.columns([1, 5], vertical_alignment="center")
    with col_logo:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=64)
    with col_text:
        st.markdown(
            f"<div style='color:{COLORS['text_muted']};font-size:0.85rem;'>"
            f"Department of Civil Engineering, MIST</div>",
            unsafe_allow_html=True,
        )
        if subtitle:
            st.markdown(
                f"<div style='color:{COLORS['navy']};font-size:1.05rem;font-weight:700;'>"
                f"{subtitle}</div>", unsafe_allow_html=True,
            )
    st.markdown("<hr>", unsafe_allow_html=True)


def back_to_home_button():
    if st.button("\u2190 Back to Home", key="back_home_btn"):
        st.switch_page("Home.py")
