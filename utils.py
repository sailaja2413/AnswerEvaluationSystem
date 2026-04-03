"""
utils.py  NLP Evaluation Utilities
Handles: TF-IDF, Cosine Similarity, Topic Extraction, Text Preprocessing
"""

import re
import string
import sqlite3
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# TEXT PREPROCESSING

def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# STOPWORDS

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "its", "their", "they", "them", "from",
    "which", "when", "where", "how", "what", "who", "as", "by",
    "also", "into", "through", "than", "more", "some", "such",
    "if", "all", "about", "one", "two", "three", "four", "five",
    "use", "used", "using"
}


def extract_keywords(text, top_n=15):
    cleaned = preprocess_text(text)
    words = cleaned.split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 3]
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    return unique_keywords[:top_n]


def compute_cosine_similarity(text1, text2):
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    if not t1 or not t2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([t1, t2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(round(sim[0][0], 4))
    except Exception:
        return 0.0


def analyze_topic_coverage(reference, student):
    ref_topics = set(extract_keywords(reference, top_n=20))
    stu_topics = set(extract_keywords(student, top_n=20))
    covered = ref_topics.intersection(stu_topics)
    missing = ref_topics.difference(stu_topics)
    extra   = stu_topics.difference(ref_topics)
    coverage_pct = (len(covered) / len(ref_topics) * 100) if ref_topics else 0
    return {
        "covered_topics": sorted(list(covered)),
        "missing_topics": sorted(list(missing)),
        "extra_topics":   sorted(list(extra)),
        "coverage_percentage": round(coverage_pct, 1),
        "ref_topic_count": len(ref_topics),
        "stu_topic_count": len(stu_topics),
    }


def nlp_evaluate(reference, student):
    similarity = compute_cosine_similarity(reference, student)
    coverage   = analyze_topic_coverage(reference, student)
    base_score = (similarity * 60) + (coverage["coverage_percentage"] * 0.40)
    base_score = min(100, max(0, base_score))
    if base_score >= 85:
        grade = "A (Excellent)"
    elif base_score >= 70:
        grade = "B (Good)"
    elif base_score >= 55:
        grade = "C (Average)"
    elif base_score >= 40:
        grade = "D (Below Average)"
    else:
        grade = "F (Needs Improvement)"
    return {
        "similarity_score": round(similarity * 100, 2),
        "nlp_score": round(base_score, 2),
        "grade": grade,
        "coverage": coverage,
        "word_count_ref": len(reference.split()),
        "word_count_stu": len(student.split()),
    }


def generate_feedback(eval_result):
    score        = eval_result.get("final_score", eval_result.get("nlp_score", 0))
    coverage     = eval_result.get("coverage", {})
    similarity   = eval_result.get("similarity_score", 0)
    missing      = coverage.get("missing_topics", [])
    covered      = coverage.get("covered_topics", [])
    coverage_pct = coverage.get("coverage_percentage", 0)
    strengths    = []
    weaknesses   = []
    suggestions  = []
    if similarity >= 70:
        strengths.append("Your answer is highly similar to the reference. Great understanding!")
    elif similarity >= 50:
        strengths.append("Your answer captures the core ideas reasonably well.")
    if coverage_pct >= 70:
        strengths.append("You covered {}% of key topics. Impressive breadth!".format(coverage_pct))
    elif coverage_pct >= 50:
        strengths.append("You mentioned {} important concepts correctly.".format(len(covered)))
    if eval_result.get("word_count_stu", 0) >= eval_result.get("word_count_ref", 1) * 0.6:
        strengths.append("Your answer has good length and detail.")
    if missing:
        weaknesses.append("Missing key topics: {}".format(", ".join(missing[:6])))
    if similarity < 40:
        weaknesses.append("Low similarity to reference. Answer may be off-topic or too brief.")
    if coverage_pct < 40:
        weaknesses.append("Only {}% topic coverage. Many concepts are left out.".format(coverage_pct))
    if missing:
        suggestions.append("Include these topics: {}".format(", ".join(missing[:6])))
    if similarity < 50:
        suggestions.append("Use more domain-specific vocabulary from your study material.")
    if eval_result.get("word_count_stu", 0) < 30:
        suggestions.append("Write a more detailed answer. Aim for at least 5-6 sentences.")
    if not weaknesses:
        suggestions.append("Keep up the great work! Review edge cases for a perfect score.")
    if not strengths:
        strengths.append("Attempt made. Keep practicing to improve.")
    return {
        "strengths":   strengths,
        "weaknesses":  weaknesses,
        "suggestions": suggestions,
        "score":       score,
    }


def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".csv"):
            import pandas as pd
            import io
            df = pd.read_csv(io.BytesIO(uploaded_file.read()))
            return " ".join(df.astype(str).values.flatten().tolist())
        elif filename.endswith(".docx"):
            import docx
            import io
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            return ""
    except Exception as e:
        return "ERROR: Could not extract text: {}".format(str(e))


# DATABASE

DB_PATH = "users.db"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'teacher')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_username TEXT NOT NULL,
            subject TEXT DEFAULT 'General',
            reference_answer TEXT,
            student_answer TEXT,
            similarity_score REAL,
            nlp_score REAL,
            ml_score REAL,
            final_score REAL,
            grade TEXT,
            covered_topics TEXT,
            missing_topics TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def register_user(username, password, role):
    if not username or not password:
        return False, "Username and password cannot be empty."
    if role not in ("student", "teacher"):
        return False, "Invalid role."
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username.strip(), hash_password(password), role))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username '{}' already exists. Choose another.".format(username)
    except Exception as e:
        return False, "Database error: {}".format(str(e))


def login_user(username, password):
    if not username or not password:
        return False, "Please enter both username and password."
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?",
                  (username.strip(), hash_password(password)))
        row = c.fetchone()
        conn.close()
        if row:
            return True, row[0]
        return False, "Invalid username or password."
    except Exception as e:
        return False, "Database error: {}".format(str(e))


def save_result(data):
    coverage = data.get("coverage", {})
    covered  = ", ".join(coverage.get("covered_topics", []))
    missing  = ", ".join(coverage.get("missing_topics", []))
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO results
            (student_username, subject, reference_answer, student_answer,
             similarity_score, nlp_score, ml_score, final_score, grade,
             covered_topics, missing_topics)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("username", "unknown"),
            data.get("subject", "General"),
            data.get("reference_answer", ""),
            data.get("student_answer", ""),
            data.get("similarity_score", 0),
            data.get("nlp_score", 0),
            data.get("ml_score", 0),
            data.get("final_score", 0),
            data.get("grade", ""),
            covered,
            missing,
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("[DB Error] save_result: {}".format(e))
        return False


def get_student_results(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT subject, similarity_score, nlp_score, ml_score,
                   final_score, grade, covered_topics, missing_topics, submitted_at
            FROM results WHERE student_username=? ORDER BY submitted_at DESC
        """, (username,))
        rows = c.fetchall()
        conn.close()
        cols = ["subject","similarity_score","nlp_score","ml_score",
                "final_score","grade","covered_topics","missing_topics","submitted_at"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def get_all_results():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT student_username, subject, similarity_score,
                   nlp_score, final_score, grade, missing_topics, submitted_at
            FROM results ORDER BY submitted_at DESC
        """)
        rows = c.fetchall()
        conn.close()
        cols = ["student","subject","similarity","nlp_score",
                "final_score","grade","missing_topics","submitted_at"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []
