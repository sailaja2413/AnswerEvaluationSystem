"""
train.py  ML Model Training Script
Dataset: dataset.csv
Model: Random Forest Regressor (+ SVR comparison)
Output: model.pkl
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (preprocess_text, extract_keywords,
                   compute_cosine_similarity, analyze_topic_coverage)

CSV_PATH   = "dataset.csv"
MODEL_PATH = "model.pkl"


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(reference, student):
    """Build 4-feature vector for a (ref, stu) pair."""
    ref_clean = preprocess_text(reference)
    stu_clean = preprocess_text(student)

    # F1: Cosine similarity
    cos_sim = compute_cosine_similarity(reference, student)

    # F2: Length ratio (capped at 2.0)
    ref_words = len(ref_clean.split()) if ref_clean else 1
    stu_words = len(stu_clean.split()) if stu_clean else 0
    length_ratio = min(stu_words / ref_words, 2.0)

    # F3: Keyword overlap ratio
    ref_kw = set(extract_keywords(reference, top_n=20))
    stu_kw = set(extract_keywords(student, top_n=20))
    kw_overlap = len(ref_kw & stu_kw) / len(ref_kw) if ref_kw else 0

    # F4: Unique word ratio in student answer
    stu_word_list = stu_clean.split()
    unique_ratio  = len(set(stu_word_list)) / len(stu_word_list) if stu_word_list else 0

    return [cos_sim, length_ratio, kw_overlap, unique_ratio]


def load_and_prepare_data(csv_path):
    """Load CSV and build feature matrix X and target y."""
    print("\n[1] Loading dataset from: {}".format(csv_path))
    df = pd.read_csv(csv_path)
    print("    Rows loaded: {}".format(len(df)))
    print("    Columns: {}".format(list(df.columns)))

    # Validate columns
    required = {"reference_answer", "student_answer", "score"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must have columns: reference_answer, student_answer, score")

    # Drop rows with nulls
    df = df.dropna(subset=list(required))
    print("    Rows after dropping nulls: {}".format(len(df)))

    print("\n[2] Building feature matrix...")
    features = []
    targets  = []
    for i, row in df.iterrows():
        feat = build_features(str(row["reference_answer"]), str(row["student_answer"]))
        features.append(feat)
        targets.append(float(row["score"]))
        if (i + 1) % 10 == 0:
            print("    Processed {}/{} rows...".format(i+1, len(df)))

    X = np.array(features)
    y = np.array(targets)
    print("    Feature matrix shape: {}".format(X.shape))
    print("    Target shape: {}".format(y.shape))
    print("    Score range: {:.1f} - {:.1f}".format(y.min(), y.max()))
    print("    Score mean:  {:.2f}".format(y.mean()))
    return X, y


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one."""

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    }

    print("\n[3] Training and comparing models...")
    print("    {:<22} {:>8} {:>8} {:>8}".format("Model", "RMSE", "MAE", "R2"))
    print("    " + "-" * 52)

    results = {}
    for name, model in candidates.items():
        # For SVR and Ridge, scale features
        if name in ("SVR", "Ridge"):
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  model)
            ])
        else:
            pipeline = Pipeline([("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.clip(y_pred, 0, 100)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results[name] = {"pipeline": pipeline, "rmse": rmse, "mae": mae, "r2": r2}
        print("    {:<22} {:>8.3f} {:>8.3f} {:>8.4f}".format(name, rmse, mae, r2))

    # Choose best by lowest RMSE
    best_name = min(results, key=lambda k: results[k]["rmse"])
    print("\n    Best model: {} (RMSE = {:.3f})".format(
        best_name, results[best_name]["rmse"]))
    return results[best_name]["pipeline"], best_name, results


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────

def save_model(pipeline, model_path):
    bundle = {"model": pipeline, "vectorizer": None}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print("\n[5] Model saved to: {}".format(model_path))


# ─────────────────────────────────────────────
# DEMO PREDICTIONS
# ─────────────────────────────────────────────

def demo_predictions(pipeline):
    print("\n[6] Demo predictions on new examples:")
    print("    " + "=" * 60)

    examples = [
        {
            "ref": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "stu": "Machine learning allows computers to learn from data automatically.",
            "label": "Partial answer",
        },
        {
            "ref": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce food in the form of glucose.",
            "stu": "Photosynthesis is when plants use sunlight, carbon dioxide, and water to produce glucose and release oxygen.",
            "label": "Very good answer",
        },
        {
            "ref": "The water cycle describes the continuous movement of water on, above, and below the Earth's surface including evaporation, condensation, and precipitation.",
            "stu": "Water goes up and comes down sometimes.",
            "label": "Poor answer",
        },
    ]

    for ex in examples:
        feat  = np.array([build_features(ex["ref"], ex["stu"])])
        score = float(np.clip(pipeline.predict(feat)[0], 0, 100))
        print("\n    Label:  {}".format(ex["label"]))
        print("    Ref:    {}...".format(ex["ref"][:70]))
        print("    Stu:    {}...".format(ex["stu"][:70]))
        print("    Score:  {:.1f}/100".format(score))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DESCRIPTIVE ANSWER EVALUATION — MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        print("[ERROR] dataset.csv not found in current directory!")
        sys.exit(1)

    # Load data
    X, y = load_and_prepare_data(CSV_PATH)

    # Train/test split
    print("\n[4] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("    Train size: {}".format(len(X_train)))
    print("    Test size:  {}".format(len(X_test)))

    # Train all models
    best_pipeline, best_name, all_results = train_all_models(
        X_train, y_train, X_test, y_test)

    # Cross-validation on full dataset
    print("\n    5-fold Cross-validation on best model ({})...".format(best_name))
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5,
                                scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores
    print("    CV RMSE scores: {}".format(
        ["{:.3f}".format(s) for s in cv_rmse]))
    print("    CV RMSE mean:   {:.3f} (+/- {:.3f})".format(
        cv_rmse.mean(), cv_rmse.std()))

    # Save model
    save_model(best_pipeline, MODEL_PATH)

    # Demo
    demo_predictions(best_pipeline)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE! model.pkl is ready.")
    print("  Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
