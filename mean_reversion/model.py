"""
Train a Logistic Regression classifier to predict next-day reversion probability
from 8 CAPM-residual features.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

TRAIN_END = "2019-12-31"
VAL_END   = "2022-12-31"

FEATURE_COLS = [
    "z_t", "z_t1", "z_t5",
    "z_roll5_mean", "z_roll5_std",
    "cum_resid_5d", "beta", "realized_vol_20d",
]


def split(features: pd.DataFrame, targets: pd.Series):
    dates = features.index.get_level_values("date")
    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)

    X_train = features.loc[train_mask, FEATURE_COLS]
    y_train = targets[train_mask]
    X_val   = features.loc[val_mask, FEATURE_COLS]
    y_val   = targets[val_mask]
    return X_train, y_train, X_val, y_val


def train(features: pd.DataFrame, targets: pd.Series) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    cache = MODEL_DIR / "best_model.pkl"
    if cache.exists():
        print("Loading cached model...")
        with open(cache, "rb") as f:
            return pickle.load(f)

    X_train, y_train, X_val, y_val = split(features, targets)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    print("Training Logistic Regression...")
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(X_train_sc, y_train)

    proba = clf.predict_proba(X_val_sc)[:, 1]
    auc = roc_auc_score(y_val, proba)
    ll  = log_loss(y_val, proba)
    acc = accuracy_score(y_val, (proba > 0.5).astype(int))
    print(f"  Val AUC={auc:.4f}  LogLoss={ll:.4f}  Acc={acc:.4f}")

    bundle = {
        "name": "LogisticRegression",
        "clf":  clf,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }
    with open(cache, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    X = X[bundle["feature_cols"]]
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    return bundle["clf"].predict_proba(X)[:, 1]


if __name__ == "__main__":
    features = pd.read_pickle(DATA_DIR / "features.pkl")
    targets  = pd.read_pickle(DATA_DIR / "targets.pkl").squeeze()
    bundle   = train(features, targets)
    print("Model saved.")
