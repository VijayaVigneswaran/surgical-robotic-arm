import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE    = os.path.join(DIR, "gesture_data.csv")
MODEL_FILE  = os.path.join(DIR, "gesture_model.pkl")
ENCODER_FILE = os.path.join(DIR, "label_encoder.pkl")


def load_data():
    print(f"Loading data from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    counts = df["gesture"].value_counts()
    print(f"\nSamples per gesture:")
    for gesture, n in counts.items():
        print(f"  {gesture:<15} {n} samples")
    print(f"  {'TOTAL':<15} {len(df)} samples\n")

    X = df.drop(columns=["gesture"]).values.astype(np.float32)
    y = df["gesture"].values
    return X, y


def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}\n")
    return y_enc, le


def train(X_train, y_train):
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, le, X_train, X_test, y_train, y_test):
    y_pred = clf.predict(X_test)

    acc_test  = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, clf.predict(X_train))

    print("=" * 52)
    print(f"  Train accuracy : {acc_train * 100:.2f}%")
    print(f"  Test  accuracy : {acc_test  * 100:.2f}%")
    print("=" * 52)

    # Confusion matrix
    labels = le.classes_
    cm = confusion_matrix(y_test, y_pred)
    col_w = max(len(l) for l in labels) + 2

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    # Per-class report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Cross-validation on full dataset
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(clf, X_all, y_all, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return acc_test


def save_artifacts(clf, le):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved   : {MODEL_FILE}")

    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(le, f)
    print(f"Encoder saved : {ENCODER_FILE}")


def main():
    # ── Load ──────────────────────────────────────────────
    X, y = load_data()
    y_enc, le = encode_labels(y)

    # ── Split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Split: {len(X_train)} train / {len(X_test)} test\n")

    # ── Train ─────────────────────────────────────────────
    clf = train(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────
    evaluate(clf, le, X_train, X_test, y_train, y_test)

    # ── Save ──────────────────────────────────────────────
    save_artifacts(clf, le)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
