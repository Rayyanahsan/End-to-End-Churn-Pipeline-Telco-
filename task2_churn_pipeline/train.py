import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# ===== Load dataset =====
df = pd.read_csv('telco_churn.csv')  # <-- Put your dataset here
target = 'Churn'
y = df[target].astype(int) if df[target].dtype != 'int' else df[target]
X = df.drop(columns=[target])

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(    transformers=[\ 
        ('num', StandardScaler(), num_cols),\ 
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)\ 
    ])

# Two candidate models
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

pipe_lr = Pipeline([('pre', pre), ('clf', logreg)])
pipe_rf = Pipeline([('pre', pre), ('clf', rf)])

param_grid_lr = {    'clf__C': [0.1, 1.0, 3.0],\ 
    'clf__penalty': ['l2']\ 
}
param_grid_rf = {    'clf__n_estimators': [200, 400],\ 
    'clf__max_depth': [None, 8, 16]\ 
}

# Train/val split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def tune(pipe, grid):\ 
    gs = GridSearchCV(pipe, grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)\ 
    gs.fit(X_train, y_train)\ 
    return gs

gs_lr = tune(pipe_lr, param_grid_lr)
gs_rf = tune(pipe_rf, param_grid_rf)

# Evaluate
def evaluate(model, name):\ 
    preds = model.predict(X_test)\ 
    acc = accuracy_score(y_test, preds)\ 
    f1 = f1_score(y_test, preds)\ 
    print(f'[{name}] ACC={acc:.4f} F1={f1:.4f}')\ 
    print(classification_report(y_test, preds))\ 
    return f1

f1_lr = evaluate(gs_lr.best_estimator_, 'LogReg')
f1_rf = evaluate(gs_rf.best_estimator_, 'RandomForest')

best = gs_lr if f1_lr >= f1_rf else gs_rf
print('Best model:', type(best.best_estimator_.named_steps['clf']).__name__)

# Save pipeline
joblib.dump(best.best_estimator_, 'churn_pipeline.joblib')
print('Saved: churn_pipeline.joblib')
