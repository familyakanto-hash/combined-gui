# =============================================================================
#  RHA-Blended Concrete ML Toolkit — Home
#  Department of Civil Engineering, Military Institute of Science and
#  Technology (MIST), Bangladesh
# =============================================================================
import os
import streamlit as st
from utils import inject_base_css, COLORS, LOGO_FILE

st.set_page_config(
    page_title="RHA Concrete ML Toolkit | CE, MIST",
    page_icon="\U0001F3DB\uFE0F",
    layout="wide",
)

inject_base_css()

if "category" not in st.session_state:
    st.session_state.category = None

# ----------------------------------------------------------------- Hero / logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=140)

st.markdown(
    f"<h1 style='text-align:center;margin-bottom:0;'>RHA-Blended Concrete "
    f"ML Toolkit</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:{COLORS['text_muted']};"
    f"font-size:1.05rem;margin-top:0.2rem;'>Department of Civil Engineering "
    f"&middot; Military Institute of Science and Technology (MIST), Bangladesh</p>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------- About blurb
with st.container():
    st.markdown(
        f"""
        <div class="category-card" style="margin-bottom:1.6rem;">
        <p style="margin-bottom:0.6rem;">
        The <b>Department of Civil Engineering at MIST</b> trains engineers to
        design and build resilient, sustainable infrastructure, with a growing
        focus on applying data-driven and computational methods alongside
        traditional structural and materials engineering. This toolkit is one
        outcome of that direction: an undergraduate thesis exploring how
        machine learning can accelerate the study of alternative, more
        sustainable construction materials.
        </p>
        <p style="margin-bottom:0;">
        <b>Rice Husk Ash (RHA)</b>, a by-product of rice milling, can partially
        replace cement in concrete &mdash; reducing both cost and the carbon
        footprint of construction, provided its effect on strength and
        durability is well understood. This app packages five machine-learning
        models, each trained on an experimental RHA-blended concrete dataset,
        into a single interactive toolkit. For any of the five properties
        below, you can either <b>predict the property from a mix design</b>,
        or <b>work backward from a target value to a set of candidate mixes</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"<h3 style='text-align:center;'>What would you like to explore?</h3>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------- Category cards
cc1, cc2 = st.columns(2, gap="large")

with cc1:
    st.markdown(
        f"""
        <div class="category-card">
            <div class="category-title">\U0001F9F1 Mechanical Properties</div>
            <div class="category-desc">
                Strength-related behaviour of the hardened concrete.
            </div>
            <span class="pill pill-mech">Compressive strength</span>
            <span class="pill pill-mech">Flexural strength</span>
            <span class="pill pill-mech">Split tensile strength</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Explore Mechanical Properties \u2192", type="primary",
                 use_container_width=True, key="btn_mech"):
        st.session_state.category = "mechanical"

with cc2:
    st.markdown(
        f"""
        <div class="category-card">
            <div class="category-title">\U0001F6E1\uFE0F Durability Properties</div>
            <div class="category-desc">
                How well the concrete resists water and chloride ingress over time.
            </div>
            <span class="pill pill-dur">Water absorption (28-day)</span>
            <span class="pill pill-dur">RCPT (28-day, chloride resistance)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Explore Durability Properties \u2192", type="primary",
                 use_container_width=True, key="btn_dur"):
        st.session_state.category = "durability"

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------- Sub-menu
if st.session_state.category == "mechanical":
    st.markdown("---")
    st.markdown("#### \U0001F9F1 Mechanical Properties \u2014 choose a property")
    m1, m2, m3 = st.columns(3)
    with m1:
        if st.button("Compressive Strength", use_container_width=True, key="go_comp"):
            st.switch_page("pages/1_Compressive_Strength.py")
    with m2:
        if st.button("Flexural Strength", use_container_width=True, key="go_flex"):
            st.switch_page("pages/2_Flexural_Strength.py")
    with m3:
        if st.button("Split Tensile Strength", use_container_width=True, key="go_split"):
            st.switch_page("pages/3_Split_Tensile_Strength.py")

elif st.session_state.category == "durability":
    st.markdown("---")
    st.markdown("#### \U0001F6E1\uFE0F Durability Properties \u2014 choose a property")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Water Absorption (28-day)", use_container_width=True, key="go_wa"):
            st.switch_page("pages/4_Water_Absorption.py")
    with d2:
        if st.button("RCPT (28-day)", use_container_width=True, key="go_rcpt"):
            st.switch_page("pages/5_RCPT.py")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center;color:{COLORS['text_muted']};font-size:0.8rem;'>"
    f"All predictions are for preliminary screening within the training domain "
    f"and do not replace experimental verification.</p>",
    unsafe_allow_html=True,
)
