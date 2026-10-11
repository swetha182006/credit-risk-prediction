"""Streamlit app: credit-risk prediction with SHAP explanations."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.explain import explain, get_explainer

ART = Path("artifacts")
st.set_page_config(page_title="Explainable Credit Risk", page_icon="💳", layout="wide")


@st.cache_resource
def load_bundle():
    bundle = joblib.load(ART / "model.joblib")
    bundle["explainer"] = get_explainer(bundle["model"])
    return bundle


def render_input_form(schema):
    values = {}
    cols = st.columns(3)
    for i, (name, spec) in enumerate(schema.items()):
        label = name.replace("_", " ").title()
        with cols[i % 3]:
            if spec["type"] == "cat":
                opts = spec["options"]
                values[name] = st.selectbox(label, opts, index=opts.index(spec["default"]))
            elif spec["is_int"]:
                values[name] = st.number_input(
                    label, min_value=int(spec["min"]), max_value=int(spec["max"] * 2),
                    value=int(spec["median"]), step=1,
                )
            else:
                values[name] = st.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"] * 2),
                    value=float(spec["median"]),
                )
    return values


st.title("💳 Explainable Credit Risk Predictor")
st.caption("Predicts the probability that an applicant is a bad credit risk, and shows *why* using SHAP.")

if not (ART / "model.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

bundle = load_bundle()
tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Model insights", "ℹ️ About"])

with tab_predict:
    with st.form("applicant"):
        values = render_input_form(bundle["schema"])
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        row = pd.DataFrame([values])[bundle["columns"]]
        Xt = bundle["preprocessor"].transform(row)
        prob = float(bundle["model"].predict_proba(Xt)[0, 1])
        is_high = prob >= bundle["threshold"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probability of bad credit risk", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            if is_high:
                st.error(f"HIGH RISK (above decision threshold {bundle['threshold']:.2f})")
            else:
                st.success(f"LOW RISK (below decision threshold {bundle['threshold']:.2f})")

        exp = explain(bundle["explainer"], Xt)
        with c2:
            st.subheader("Why this prediction?")
            plt.figure()
            import shap

            shap.plots.waterfall(exp[0], max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("Red bars push the applicant toward HIGH risk, blue bars toward LOW risk.")

        contrib = pd.Series(exp.values[0], index=Xt.columns).sort_values(key=abs, ascending=False).head(3)
        st.markdown("**Top 3 drivers:** " + ", ".join(
            f"`{k}` ({'↑ risk' if v > 0 else '↓ risk'})" for k, v in contrib.items()
        ))

with tab_insights:
    mpath = ART / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.subheader(f"Selected model: {m['selected_model']}")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, m["test"].items()):
            col.metric(k.replace("_", " ").upper(), v)
        st.markdown("**Cross-validated ROC-AUC (training set)**")
        st.dataframe(pd.Series(m["cv_roc_auc"], name="ROC-AUC").to_frame())
    left, right = st.columns(2)
    for col, (img, title) in zip(
        (left, right), [("shap_bar.png", "Global feature importance"), ("shap_beeswarm.png", "Feature impact (SHAP)")]
    ):
        if (ART / img).exists():
            col.subheader(title)
            col.image(str(ART / img))
    left, right = st.columns(2)
    for col, (img, title) in zip((left, right), [("roc_curve.png", "ROC curve"), ("confusion_matrix.png", "Confusion matrix")]):
        if (ART / img).exists():
            col.subheader(title)
            col.image(str(ART / img))

with tab_about:
    st.markdown(
        """
**Pipeline:** raw features → imputation + one-hot encoding → tuned tree ensemble
(Random Forest vs Gradient Boosting, compared to a Logistic Regression baseline) → F1-optimised decision threshold → SHAP explanations.

**Data:** German Credit (OpenML `credit-g`), 1,000 applicants.

**Caveat:** educational project on a small public dataset, not for real lending decisions.
"""
    )
