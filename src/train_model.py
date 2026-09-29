import json
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import DATASET_PATH, FEATURES, MODEL_DIR


def load_and_clean():
    df = pd.read_csv(DATASET_PATH)
    before = len(df)
    df = df.drop_duplicates().dropna()
    print(f"Loaded {before} rows, {len(df)} after removing duplicates/missing values")
    return df


def train():
    df = load_and_clean()
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=60, max_depth=3, random_state=42
        ),
    }

    best_name, best_model, best_acc = None, None, -1
    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="macro", zero_division=0)
        rec = recall_score(y_test, pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        scores[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        print(f"{name:20s}  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")
    print(classification_report(y_test, best_model.predict(X_test), target_names=le.classes_))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "features": FEATURES,
                "accuracy": best_acc,
                "all_models": scores,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model, label encoder and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    train()
