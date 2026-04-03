"""
model.py  ML Model Wrapper for Answer Evaluation
Loads saved model.pkl and provides predict() interface.
Falls back to NLP score if model not trained yet.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "model.pkl"


class AnswerEvaluator:
    """Wraps the trained ML model with a clean predict interface."""

    def __init__(self):
        self.model      = None
        self.vectorizer = None
        self.is_trained = False
        self._load()

    def _load(self):
        """Load model and vectorizer from disk if they exist."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    bundle = pickle.load(f)
                self.model      = bundle["model"]
                self.vectorizer = bundle["vectorizer"]
                self.is_trained = True
                print("[Model] Loaded model.pkl successfully.")
            except Exception as e:
                print("[Model] Failed to load model.pkl: {}".format(e))
                self.is_trained = False
        else:
            print("[Model] model.pkl not found. Run train.py first.")

    def _build_features(self, reference, student):
        """
        Build feature vector for a (reference, student) pair.
        Features:
          0: cosine similarity (TF-IDF)
          1: length ratio (student / reference word count)
          2: keyword overlap ratio
          3: unique word ratio in student answer
        """
        from utils import preprocess_text, extract_keywords, compute_cosine_similarity

        ref_clean = preprocess_text(reference)
        stu_clean = preprocess_text(student)

        # Feature 1: Cosine similarity
        cos_sim = compute_cosine_similarity(reference, student)

        # Feature 2: Length ratio
        ref_words = len(ref_clean.split()) if ref_clean else 1
        stu_words = len(stu_clean.split()) if stu_clean else 0
        length_ratio = min(stu_words / ref_words, 2.0)

        # Feature 3: Keyword overlap
        ref_kw = set(extract_keywords(reference, top_n=20))
        stu_kw = set(extract_keywords(student, top_n=20))
        kw_overlap = len(ref_kw & stu_kw) / len(ref_kw) if ref_kw else 0

        # Feature 4: Unique word ratio in student
        stu_word_list = stu_clean.split()
        unique_ratio  = len(set(stu_word_list)) / len(stu_word_list) if stu_word_list else 0

        return np.array([[cos_sim, length_ratio, kw_overlap, unique_ratio]])

    def predict(self, reference, student):
        """
        Predict score for student answer vs reference.
        Returns predicted score (0-100).
        If model not loaded, falls back to NLP-based score.
        """
        if not self.is_trained:
            # Fallback: use cosine similarity scaled to 0-100
            from utils import compute_cosine_similarity, analyze_topic_coverage
            sim  = compute_cosine_similarity(reference, student)
            cov  = analyze_topic_coverage(reference, student)
            score = (sim * 60) + (cov["coverage_percentage"] * 0.40)
            return round(min(100, max(0, score)), 2)

        try:
            features = self._build_features(reference, student)
            predicted = self.model.predict(features)[0]
            return round(float(np.clip(predicted, 0, 100)), 2)
        except Exception as e:
            print("[Model] Predict error: {}".format(e))
            return 50.0


# Singleton instance (loaded once)
_evaluator = None


def get_evaluator():
    """Return singleton AnswerEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = AnswerEvaluator()
    return _evaluator


def predict_score(reference, student):
    """Convenience function: predict score directly."""
    evaluator = get_evaluator()
    return evaluator.predict(reference, student)
