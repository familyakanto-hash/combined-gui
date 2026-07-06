# =============================================================================
#  RHA-Blended Concrete — Flexural Strength Toolkit
#  Tab 1: Predict strength  (forward)  — mix in  -> strength out
#  Tab 2: Design a mix      (inverse)  — target strength in -> best mixes out
#
#  Best model from the thesis: XGBoost (Optuna-tuned) on a leakage-free pipeline
#     median impute -> IQR winsorise (x1.5) -> Yeo-Johnson -> RobustScaler.
#  Run:  streamlit run flexural_app.py
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import inject_base_css, show_header, back_to_home_button

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------- config
SEED = 42
TARGET = "flexural_strength"
FEATURES = ["rha_replacement_pct", "water_binder_ratio", "cement_kg_m3",
            "fine_aggregate", "coarse_aggregate"]
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dataset_flexural_strength.csv")

LABELS = {
    "rha_replacement_pct": "RHA replacement (%)",
    "water_binder_ratio":  "Water / binder ratio",
    "cement_kg_m3":        "Cement (kg/m\u00b3)",
    "fine_aggregate":      "Fine aggregate (kg/m\u00b3)",
    "coarse_aggregate":    "Coarse aggregate (kg/m\u00b3)",
}
# min, max, default, step  (physically valid ranges from the notebook)
RANGES = {
    "rha_replacement_pct": (0.0, 80.0, 10.0, 1.0),
    "water_binder_ratio":  (0.10, 1.00, 0.45, 0.01),
    "cement_kg_m3":        (100.0, 700.0, 369.0, 5.0),
    "fine_aggregate":      (100.0, 1200.0, 680.0, 5.0),
    "coarse_aggregate":    (0.0, 1600.0, 1102.0, 5.0),
}

# Best hyperparameters found via Optuna (Step 8 of the notebook)
XGB_PARAMS = dict(n_estimators=132, max_depth=4, learning_rate=0.2993232251771097,
                  subsample=0.7680076051373114, colsample_bytree=0.7219847221934368,
                  reg_alpha=0.011581574867011515, reg_lambda=7.413801315981653,
                  random_state=SEED, verbosity=0, eval_metric="rmse")
N_CANDIDATES = 40000


# ----------------------------------------------------------------------------- pipeline
class WinsorClip(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i, c in enumerate(X.columns):
            X[c] = X[c].clip(self.lower_.iloc[i], self.upper_.iloc[i])
        return X.values


def build_pipeline(model):
    return Pipeline([
        ("imputer",     SimpleImputer(strategy="median")),
        ("winsor",      WinsorClip(1.5)),
        ("transformer", PowerTransformer(method="yeo-johnson")),
        ("scaler",      RobustScaler()),
        ("model",       model),
    ])


@st.cache_resource(show_spinner="Training the model and preparing the toolkit "
                                 "(first run only)\u2026")
def setup():
    """Train XGBoost once; build candidate pool for the inverse tab."""
    df = pd.read_csv(DATA_FILE)[FEATURES + [TARGET]].drop_duplicates()
    df = df[df[TARGET].notna()].reset_index(drop=True)

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = build_pipeline(XGBRegressor(**XGB_PARAMS))
    pipe.fit(X_tr, y_tr)
    pred_te = pipe.predict(X_te)
    metrics = {"R2": r2_score(y_te, pred_te),
               "RMSE": mean_squared_error(y_te, pred_te) ** 0.5,
               "MAE": mean_absolute_error(y_te, pred_te),
               "n": len(df)}

    # extrapolation bounds (1st-99th pct of training data) for the forward tab
    bounds = {c: (X_tr[c].quantile(0.01), X_tr[c].quantile(0.99)) for c in FEATURES}

    # candidate pool for the inverse tab
    rng = np.random.default_rng(SEED)
    p1 = X_tr.quantile(0.01); p99 = X_tr.quantile(0.99); sd = X_tr.std()
    base = X_tr.sample(N_CANDIDATES, replace=True, random_state=SEED).reset_index(drop=True)
    noise = rng.normal(0, 1, (N_CANDIDATES, len(FEATURES))) * (0.4 * sd.values)
    cand = np.clip(base.values + noise, p1.values, p99.values)
    pool = pd.DataFrame(np.vstack([X_tr.values, cand]), columns=FEATURES)
    pool["predicted_strength"] = pipe.predict(pool[FEATURES])

    return pipe, metrics, bounds, pool, sd


# ----------------------------------------------------------------------------- UI
st.set_page_config(page_title="Flexural Strength | RHA Concrete Toolkit",
                   page_icon="\U0001F9F1", layout="centered")
inject_base_css()
back_to_home_button()

st.title("\U0001F9F1 Flexural Strength")
st.caption("Best model from the analysis: **XGBoost (Optuna-tuned)** (leakage-free "
           "pipeline). Predict the flexural strength of a mix, or design a mix "
           "for a target strength.")

pipe, metrics, bounds, pool, sd = setup()

with st.expander("Model performance (held-out test set)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R\u00b2", f"{metrics['R2']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f} MPa")
    c3.metric("MAE", f"{metrics['MAE']:.2f} MPa")
    c4.metric("Samples", f"{metrics['n']}")
    st.caption("80/20 split, seed 42 \u2014 consistent with the thesis methodology.")

tab_predict, tab_design = st.tabs(["\U0001F50E  Predict strength", "\U0001F9EE  Design a mix"])

# ============================================================ TAB 1: forward
with tab_predict:
    st.subheader("Enter your mix design")
    col1, col2 = st.columns(2)
    vals = {}
    for i, f in enumerate(FEATURES):
        lo, hi, default, step = RANGES[f]
        target_col = col1 if i % 2 == 0 else col2
        vals[f] = target_col.number_input(
            LABELS[f], min_value=float(lo), max_value=float(hi),
            value=float(default), step=float(step),
            format="%.2f" if f == "water_binder_ratio" else "%.1f",
            key=f"fwd_{f}",
        )

    if st.button("Calculate flexural strength", type="primary",
                 use_container_width=True, key="btn_predict"):
        X_user = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
        pred = float(pipe.predict(X_user)[0])

        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.95rem;color:#666'>Predicted flexural strength</div>"
            f"<div style='font-size:3.2rem;font-weight:700;color:#1a7f5a;line-height:1.1'>"
            f"{pred:.2f} <span style='font-size:1.4rem'>MPa</span></div>"
            f"<div style='font-size:0.8rem;color:#888'>approx. \u00b1{metrics['RMSE']:.2f} "
            f"MPa (model RMSE)</div></div>", unsafe_allow_html=True)

        out = [LABELS[f] for f in FEATURES if not (bounds[f][0] <= vals[f] <= bounds[f][1])]
        if out:
            st.warning("Outside the bulk of the training data: " + ", ".join(out)
                       + ". The model is interpolative \u2014 treat with caution and "
                       "verify with testing for critical use.")
        else:
            st.success("All inputs lie within the training range \u2014 prediction is "
                       "interpolative.")

# ============================================================ TAB 2: inverse
with tab_design:
    st.subheader("What strength do you need?")
    s_min = float(pool["predicted_strength"].quantile(0.01))
    s_max = float(pool["predicted_strength"].quantile(0.99))

    target = st.number_input(
        "Target flexural strength (MPa)",
        min_value=round(s_min, 1), max_value=round(s_max, 1),
        value=float(np.clip(4.5, s_min, s_max)), step=0.1,
        help=f"Achievable range for this dataset is roughly {s_min:.1f}\u2013{s_max:.1f} MPa.",
        key="inv_target",
    )

    c1, c2 = st.columns(2)
    objective = c1.selectbox(
        "What does \u201cbest\u201d mean?",
        ["Most sustainable (max RHA, least cement)",
         "Maximum RHA replacement",
         "Lowest cement content",
         "Closest match to target"],
        key="inv_obj",
    )
    tol = c2.slider("Strength tolerance (\u00b1 MPa)", 0.1, 1.0, 0.3, 0.05, key="inv_tol")
    k = st.slider("How many mix options to show", 3, 12, 6, key="inv_k")

    if st.button("Find best mixes", type="primary", use_container_width=True,
                 key="btn_design"):
        hits = pool[(pool["predicted_strength"] - target).abs() <= tol].copy()
        note = None
        if hits.empty:
            hits = pool.iloc[(pool["predicted_strength"] - target).abs()
                             .argsort()[:300]].copy()
            note = (f"No mix landed within \u00b1{tol:.2f} MPa of {target:.1f} MPa \u2014 "
                    "showing the closest achievable mixes instead.")

        if objective == "Most sustainable (max RHA, least cement)":
            hits = hits.sort_values(["rha_replacement_pct", "cement_kg_m3"],
                                    ascending=[False, True])
        elif objective == "Maximum RHA replacement":
            hits = hits.sort_values("rha_replacement_pct", ascending=False)
        elif objective == "Lowest cement content":
            hits = hits.sort_values("cement_kg_m3", ascending=True)
        else:
            hits = hits.reindex((hits["predicted_strength"] - target).abs()
                                .sort_values().index)

        picked = []
        for _, r in hits.iterrows():
            rv = r[FEATURES].values
            if all((np.abs(rv - p[:-1]) / sd.values).sum() > 1.0 for p in picked):
                picked.append(np.append(rv, r["predicted_strength"]))
                if len(picked) >= k:
                    break

        result = pd.DataFrame(picked, columns=FEATURES + ["predicted_strength"])
        result.insert(0, "Option", [f"Mix {i+1}" for i in range(len(result))])

        if note:
            st.warning(note)

        st.markdown("#### Recommended mixes")
        show = result.copy()
        show.columns = (["Option"] + [LABELS[f] for f in FEATURES]
                        + ["Predicted strength (MPa)"])
        st.dataframe(
            show.style.format({c: "{:.2f}" for c in show.columns if c != "Option"}),
            use_container_width=True, hide_index=True,
        )

        best = result.iloc[0]
        st.success(
            f"Top pick \u2014 **{best['rha_replacement_pct']:.0f}% RHA**, "
            f"w/b **{best['water_binder_ratio']:.2f}**, "
            f"cement **{best['cement_kg_m3']:.0f}**, "
            f"fine agg **{best['fine_aggregate']:.0f}**, "
            f"coarse agg **{best['coarse_aggregate']:.0f}** kg/m\u00b3 "
            f"\u2192 predicted **{best['predicted_strength']:.2f} MPa**."
        )
        st.download_button(
            "Download these mixes (CSV)",
            result.to_csv(index=False).encode(),
            file_name=f"flexural_mixes_{target:.1f}MPa.csv", mime="text/csv",
            key="dl_design",
        )

st.markdown("---")
st.caption("Predictions are for preliminary screening within the training domain and do "
           "not replace experimental verification. Several mixes can reach the same "
           "strength, so the designer returns a diverse ranked set, not a single answer. "
           "Built from the thesis: *Machine-Learning Analysis of Mechanical and Durability "
           "Properties of Rice-Husk-Ash Blended Concrete.*")
