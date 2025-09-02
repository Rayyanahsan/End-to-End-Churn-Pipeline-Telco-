# Task 2 — End-to-End ML Pipeline (Telco Churn)

- Encodes categoricals, scales numerics via `ColumnTransformer`
- Uses `Pipeline` with Logistic Regression and Random Forest
- Hyperparameter tuning via `GridSearchCV`
- Saves final pipeline with `joblib`

Run:
```bash
pip install -r ../requirements.txt
python train.py
```
Dataset: place `telco_churn.csv` in this folder.
