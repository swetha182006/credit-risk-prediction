# 💳 Explainable Credit Risk Predictor

End-to-end ML project: data → preprocessing → model comparison & tuning → evaluation → **SHAP explainability** → **Streamlit app**.

Given an applicant's profile, the app predicts the probability of being a bad credit risk and explains the prediction feature by feature.

## Project structure
```
app.py              Streamlit app (predict + SHAP waterfall + model insights)
src/data.py         Data loading (German Credit from OpenML, or your own CSV)
src/features.py     Preprocessing pipeline + input schema for the app form
src/train.py        CV, hyperparameter tuning, threshold selection, evaluation, SHAP plots
src/explain.py      SHAP helpers
artifacts/          Saved model, metrics and plots (commit these for deployment)
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # trains and writes artifacts/ (needs internet once for OpenML)
streamlit run app.py
```

## Method
- **Split:** 80/20 stratified; all tuning uses 5-fold stratified CV on the training set only (no leakage: preprocessing lives inside the CV pipeline).
- **Models:** Logistic Regression baseline vs tuned Random Forest and Gradient Boosting (RandomizedSearchCV, ROC-AUC).
- **Threshold:** chosen from out-of-fold predictions to maximise F1 on the "bad" class, since the classes are imbalanced (~30% bad).
- **Explainability:** SHAP `TreeExplainer` – global (bar + beeswarm) and per-applicant (waterfall).


## Deploy on Streamlit Community Cloud (free)
1. Run `python -m src.train`, then push the whole folder (including `artifacts/`) to a public GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick the repo, branch `main`, main file `app.py`, and deploy.
4. Paste the live URL at the top of this README and on your resume.

## Ideas to extend
- Add a LightGBM/XGBoost model and compare
- Probability calibration (`CalibratedClassifierCV`) and a calibration plot
- Fairness check across `age` or `personal_status` groups
- Dockerfile + FastAPI endpoint for serving
