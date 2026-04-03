"""
app.py  Descriptive Answer Evaluation System
Streamlit Multi-page App with Authentication

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Internal modules
from utils import (
    init_db, register_user, login_user, save_result,
    get_student_results, get_all_results, nlp_evaluate,
    generate_feedback, extract_text_from_file
)
from model import predict_score


# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Answer Evaluation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database on startup
init_db()


# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-left: 5px solid #4f8ef7;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .score-excellent { color: #22c55e; font-weight: bold; font-size: 2rem; }
    .score-good      { color: #3b82f6; font-weight: bold; font-size: 2rem; }
    .score-average   { color: #f59e0b; font-weight: bold; font-size: 2rem; }
    .score-poor      { color: #ef4444; font-weight: bold; font-size: 2rem; }
    .topic-chip-green {
        display: inline-block;
        background: #dcfce7; color: #166534;
        border-radius: 20px; padding: 2px 12px;
        margin: 3px; font-size: 0.85rem;
    }
    .topic-chip-red {
        display: inline-block;
        background: #fee2e2; color: #991b1b;
        border-radius: 20px; padding: 2px 12px;
        margin: 3px; font-size: 0.85rem;
    }
    .feedback-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────

def session_defaults():
    defaults = {
        "logged_in":   False,
        "username":    None,
        "role":        None,
        "eval_result": None,
        "page":        "home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

session_defaults()


# ─────────────────────────────────────────────────────────────
# HELPER: SCORE COLOR CLASS
# ─────────────────────────────────────────────────────────────

def score_color_class(score):
    if score >= 85: return "score-excellent"
    if score >= 70: return "score-good"
    if score >= 50: return "score-average"
    return "score-poor"


# ─────────────────────────────────────────────────────────────
# PAGE 1: AUTHENTICATION (LOGIN / SIGNUP)
# ─────────────────────────────────────────────────────────────

def page_auth():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Descriptive Answer Evaluation System</h1>
        <p>AI-powered evaluation using NLP & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

        # ── LOGIN TAB ──
        with tab_login:
            st.subheader("Welcome Back!")
            username = st.text_input("Username", key="login_user",
                                     placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass",
                                     placeholder="Enter your password")

            if st.button("Login", use_container_width=True, key="btn_login"):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.role      = result  # 'student' or 'teacher'
                        st.session_state.page      = "dashboard"
                        st.success("Welcome, {}! ({})".format(username, result.title()))
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please fill in all fields.")

            st.markdown("---")
            st.info("**Demo accounts** — create via Sign Up tab\n\nOr use: `student1 / pass123` (after signup)")

        # ── SIGNUP TAB ──
        with tab_signup:
            st.subheader("Create Account")
            new_user = st.text_input("Choose Username", key="reg_user",
                                     placeholder="e.g. john_doe")
            new_pass = st.text_input("Choose Password", type="password", key="reg_pass",
                                     placeholder="At least 4 characters")
            conf_pass = st.text_input("Confirm Password", type="password", key="reg_conf",
                                      placeholder="Repeat password")
            role = st.selectbox("I am a:", ["student", "teacher"], key="reg_role")

            if st.button("Create Account", use_container_width=True, key="btn_signup"):
                if not new_user or not new_pass:
                    st.error("All fields are required.")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                elif new_pass != conf_pass:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_user, new_pass, role)
                    if ok:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎓 Navigation")
        st.markdown("---")
        st.markdown("👤 **User:** `{}`".format(st.session_state.username))
        st.markdown("🏷️ **Role:** `{}`".format(st.session_state.role.title()))
        st.markdown("---")

        # Student navigation
        if st.session_state.role == "student":
            if st.button("🏠 Dashboard",    use_container_width=True): st.session_state.page = "dashboard";  st.rerun()
            if st.button("📤 Submit Answer", use_container_width=True): st.session_state.page = "submit";    st.rerun()
            if st.button("📊 My Results",    use_container_width=True): st.session_state.page = "results";   st.rerun()
            if st.button("💬 Feedback",      use_container_width=True): st.session_state.page = "feedback";  st.rerun()

        # Teacher navigation
        if st.session_state.role == "teacher":
            if st.button("🏠 Dashboard",      use_container_width=True): st.session_state.page = "dashboard";  st.rerun()
            if st.button("📊 All Results",    use_container_width=True): st.session_state.page = "results";    st.rerun()
            if st.button("📈 Teacher Panel",  use_container_width=True): st.session_state.page = "teacher";   st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # Model status
        st.markdown("---")
        model_exists = os.path.exists("model.pkl")
        if model_exists:
            st.success("✅ ML Model: Loaded")
        else:
            st.warning("⚠️ ML Model: Not trained\nRun `python train.py`")


# ─────────────────────────────────────────────────────────────
# PAGE 2: STUDENT DASHBOARD
# ─────────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown("## 🏠 Welcome, {}!".format(st.session_state.username))

    results = get_student_results(st.session_state.username) \
              if st.session_state.role == "student" else []

    col1, col2, col3, col4 = st.columns(4)

    total       = len(results)
    avg_score   = round(sum(r["final_score"] or 0 for r in results) / total, 1) if total else 0
    best_score  = max((r["final_score"] or 0 for r in results), default=0)
    pass_count  = sum(1 for r in results if (r["final_score"] or 0) >= 55)

    col1.metric("📝 Total Submissions", total)
    col2.metric("📊 Average Score",     "{}%".format(avg_score))
    col3.metric("🏆 Best Score",        "{}%".format(round(best_score, 1)))
    col4.metric("✅ Passed",            "{}/{}".format(pass_count, total))

    st.markdown("---")

    if results:
        st.subheader("📋 Recent Submissions")
        df = pd.DataFrame(results[:5])
        show_cols = ["subject", "final_score", "grade", "submitted_at"]
        available = [c for c in show_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)
    else:
        st.info("No submissions yet. Go to **Submit Answer** to start!")
        if st.button("📤 Submit Your First Answer"):
            st.session_state.page = "submit"
            st.rerun()

    # Quick action buttons
    st.markdown("---")
    st.subheader("Quick Actions")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📤 Submit New Answer", use_container_width=True):
            st.session_state.page = "submit"
            st.rerun()
    with c2:
        if st.button("💬 View Feedback", use_container_width=True):
            st.session_state.page = "feedback"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE 3: SUBMIT ANSWER (Upload & Evaluate)
# ─────────────────────────────────────────────────────────────

def page_submit():
    st.markdown("## 📤 Submit Answer for Evaluation")
    st.markdown("Enter answers directly or upload files (.txt, .csv, .docx)")

    # Subject
    subject = st.text_input("📚 Subject / Topic Name",
                            placeholder="e.g. Machine Learning, Biology, History",
                            value="General")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # REFERENCE ANSWER
    with col1:
        st.subheader("📖 Reference Answer")
        ref_input_method = st.radio("Input method:", ["Type text", "Upload file"],
                                    key="ref_method", horizontal=True)
        reference_text = ""
        if ref_input_method == "Type text":
            reference_text = st.text_area(
                "Reference Answer",
                placeholder="Paste the model/reference answer here...",
                height=200,
                key="ref_text",
                label_visibility="collapsed"
            )
        else:
            ref_file = st.file_uploader("Upload Reference", type=["txt", "csv", "docx"],
                                        key="ref_file")
            if ref_file:
                reference_text = extract_text_from_file(ref_file)
                if reference_text.startswith("ERROR"):
                    st.error(reference_text)
                    reference_text = ""
                else:
                    st.success("✅ Extracted {} words".format(len(reference_text.split())))
                    st.text_area("Preview:", reference_text[:500] + "...",
                                 height=100, disabled=True, key="ref_preview")

    # STUDENT ANSWER
    with col2:
        st.subheader("✍️ Student Answer")
        stu_input_method = st.radio("Input method:", ["Type text", "Upload file"],
                                    key="stu_method", horizontal=True)
        student_text = ""
        if stu_input_method == "Type text":
            student_text = st.text_area(
                "Student Answer",
                placeholder="Paste the student's answer here...",
                height=200,
                key="stu_text",
                label_visibility="collapsed"
            )
        else:
            stu_file = st.file_uploader("Upload Student Answer", type=["txt", "csv", "docx"],
                                        key="stu_file")
            if stu_file:
                student_text = extract_text_from_file(stu_file)
                if student_text.startswith("ERROR"):
                    st.error(student_text)
                    student_text = ""
                else:
                    st.success("✅ Extracted {} words".format(len(student_text.split())))
                    st.text_area("Preview:", student_text[:500] + "...",
                                 height=100, disabled=True, key="stu_preview")

    st.markdown("---")

    # EVALUATE BUTTON
    if st.button("🚀 Evaluate Answer", use_container_width=True, type="primary"):
        if not reference_text or not reference_text.strip():
            st.error("❌ Reference answer is required!")
            return
        if not student_text or not student_text.strip():
            st.error("❌ Student answer is required!")
            return

        with st.spinner("🔍 Analyzing answer using NLP + ML..."):
            # NLP evaluation
            nlp_result = nlp_evaluate(reference_text, student_text)

            # ML prediction
            ml_score = predict_score(reference_text, student_text)

            # Combined final score: 50% NLP + 50% ML
            final_score = round((nlp_result["nlp_score"] * 0.5) + (ml_score * 0.5), 2)

            # Determine grade
            if final_score >= 85:   grade = "A (Excellent)"
            elif final_score >= 70: grade = "B (Good)"
            elif final_score >= 55: grade = "C (Average)"
            elif final_score >= 40: grade = "D (Below Average)"
            else:                   grade = "F (Needs Improvement)"

            # Build full result
            full_result = {
                **nlp_result,
                "ml_score":         round(ml_score, 2),
                "final_score":      final_score,
                "grade":            grade,
                "username":         st.session_state.username,
                "subject":          subject,
                "reference_answer": reference_text,
                "student_answer":   student_text,
            }

            # Save to DB
            save_result(full_result)

            # Store in session for feedback page
            st.session_state.eval_result = full_result

        st.success("✅ Evaluation complete! Scroll down to see results.")
        st.markdown("---")

        # ── SHOW RESULTS INLINE ──
        page_performance_summary(full_result)

        # Navigate buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 View Full Feedback", use_container_width=True):
                st.session_state.page = "feedback"
                st.rerun()
        with c2:
            if st.button("📤 Submit Another", use_container_width=True):
                st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE 4: PERFORMANCE SUMMARY
# ─────────────────────────────────────────────────────────────

def page_performance_summary(result=None):
    if result is None:
        result = st.session_state.get("eval_result")

    if not result:
        st.warning("No evaluation data found. Please submit an answer first.")
        if st.button("📤 Go to Submit"):
            st.session_state.page = "submit"
            st.rerun()
        return

    st.markdown("## 📊 Performance Summary")

    final   = result.get("final_score", 0)
    nlp_sc  = result.get("nlp_score", 0)
    ml_sc   = result.get("ml_score", 0)
    sim_sc  = result.get("similarity_score", 0)
    grade   = result.get("grade", "")
    subject = result.get("subject", "General")
    coverage = result.get("coverage", {})

    # Score display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Final Score",      "{:.1f}%".format(final))
    col2.metric("🧠 NLP Score",        "{:.1f}%".format(nlp_sc))
    col3.metric("🤖 ML Score",         "{:.1f}%".format(ml_sc))
    col4.metric("🔗 Similarity",       "{:.1f}%".format(sim_sc))

    st.markdown("---")

    # Grade banner
    color_class = score_color_class(final)
    st.markdown("""
    <div style="text-align:center; padding:1rem;
                background:#f8fafc; border-radius:10px; margin-bottom:1rem;">
        <div class="{color}">{score:.1f} / 100</div>
        <div style="font-size:1.3rem; font-weight:600; margin-top:0.3rem;">
            Grade: {grade}
        </div>
        <div style="color:#64748b;">Subject: {subject}</div>
    </div>
    """.format(
        color=color_class, score=final, grade=grade, subject=subject
    ), unsafe_allow_html=True)

    # Topic coverage
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Topics Covered")
        covered = coverage.get("covered_topics", [])
        if covered:
            chips = " ".join(
                '<span class="topic-chip-green">{}</span>'.format(t) for t in covered)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("No matching keywords found.")

        cov_pct = coverage.get("coverage_percentage", 0)
        st.progress(int(cov_pct) / 100,
                    text="Topic Coverage: {:.1f}%".format(cov_pct))

    with col_b:
        st.subheader("❌ Missing Topics")
        missing = coverage.get("missing_topics", [])
        if missing:
            chips = " ".join(
                '<span class="topic-chip-red">{}</span>'.format(t) for t in missing)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("🎉 All key topics covered!")

    # Word count comparison
    st.markdown("---")
    st.subheader("📝 Answer Statistics")
    wc_ref = result.get("word_count_ref", 0)
    wc_stu = result.get("word_count_stu", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Reference Words", wc_ref)
    c2.metric("Your Words",      wc_stu)
    c3.metric("Coverage %",      "{:.1f}%".format(cov_pct))


# ─────────────────────────────────────────────────────────────
# PAGE 5: STUDENT FEEDBACK
# ─────────────────────────────────────────────────────────────

def page_feedback():
    st.markdown("## 💬 Personalized Feedback")

    result = st.session_state.get("eval_result")
    if not result:
        # Try to load last result from DB
        results = get_student_results(st.session_state.username)
        if results:
            last = results[0]
            # Reconstruct minimal eval_result for feedback
            result = {
                "final_score":    last.get("final_score", 0),
                "nlp_score":      last.get("nlp_score", 0),
                "ml_score":       last.get("ml_score", 0),
                "similarity_score": last.get("similarity_score", 0),
                "grade":          last.get("grade", ""),
                "subject":        last.get("subject", "General"),
                "coverage": {
                    "covered_topics": [t.strip() for t in (last.get("covered_topics") or "").split(",") if t.strip()],
                    "missing_topics": [t.strip() for t in (last.get("missing_topics") or "").split(",") if t.strip()],
                    "coverage_percentage": 0,
                },
                "word_count_ref": 0,
                "word_count_stu": 0,
            }
        else:
            st.warning("No evaluation found. Submit an answer first!")
            if st.button("📤 Submit Answer"):
                st.session_state.page = "submit"
                st.rerun()
            return

    feedback = generate_feedback(result)

    # Summary score
    score = result.get("final_score", 0)
    grade = result.get("grade", "")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);
                padding:1.2rem; border-radius:10px; color:white; text-align:center;
                margin-bottom:1.5rem;">
        <h2 style="margin:0;">{:.1f}/100 — {}</h2>
        <p style="margin:0.3rem 0 0;">Subject: {}</p>
    </div>
    """.format(score, grade, result.get("subject", "General")), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Strengths
    with col1:
        st.subheader("💪 Strengths")
        strengths = feedback.get("strengths", [])
        if strengths:
            for s in strengths:
                st.markdown("""
                <div class="feedback-box">✅ {}</div>
                """.format(s), unsafe_allow_html=True)
        else:
            st.info("Keep working — strengths will appear here after more practice.")

    # Weaknesses
    with col2:
        st.subheader("⚠️ Areas to Improve")
        weaknesses = feedback.get("weaknesses", [])
        if weaknesses:
            for w in weaknesses:
                st.markdown("""
                <div class="feedback-box" style="border-left:4px solid #ef4444;">
                    ❌ {}
                </div>
                """.format(w), unsafe_allow_html=True)
        else:
            st.success("No major weaknesses found — great job!")

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Study Suggestions")
    suggestions = feedback.get("suggestions", [])
    for i, s in enumerate(suggestions, 1):
        st.markdown("{}. {}".format(i, s))

    # Missing topics study guide
    missing = result.get("coverage", {}).get("missing_topics", [])
    if missing:
        st.markdown("---")
        st.subheader("📚 Topics to Review")
        st.markdown("Focus on these concepts in your next study session:")
        cols = st.columns(min(len(missing), 4))
        for i, topic in enumerate(missing[:8]):
            cols[i % 4].markdown("""
            <div style="background:#fef3c7; border-radius:8px;
                        padding:0.6rem; text-align:center; margin:0.2rem;
                        font-weight:600; color:#92400e;">
                📖 {}
            </div>
            """.format(topic), unsafe_allow_html=True)

    # Action buttons
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📤 Submit Another Answer", use_container_width=True):
            st.session_state.page = "submit"
            st.rerun()
    with c2:
        if st.button("📊 View All My Results", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE 6: RESULTS HISTORY
# ─────────────────────────────────────────────────────────────

def page_results():
    role = st.session_state.role

    if role == "student":
        st.markdown("## 📊 My Evaluation History")
        results = get_student_results(st.session_state.username)
        label = "student"
    else:
        st.markdown("## 📊 All Student Results")
        results = get_all_results()
        label = "teacher"

    if not results:
        st.info("No results found yet.")
        return

    df = pd.DataFrame(results)

    # Summary stats
    if "final_score" in df.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Evaluations", len(df))
        col2.metric("Average Score",     "{:.1f}%".format(df["final_score"].mean()))
        col3.metric("Highest Score",     "{:.1f}%".format(df["final_score"].max()))

    st.markdown("---")
    st.subheader("📋 Detailed Results")

    # Select display columns dynamically
    desired_cols = ["student","subject","final_score","grade",
                    "similarity","nlp_score","submitted_at"]
    show = [c for c in desired_cols if c in df.columns]

    # Rename for display
    rename = {
        "final_score": "Final Score (%)",
        "similarity":  "Similarity (%)",
        "nlp_score":   "NLP Score (%)",
        "submitted_at": "Date",
    }
    display_df = df[show].rename(columns=rename)
    st.dataframe(display_df, use_container_width=True)

    # Score distribution chart (for teacher)
    if role == "teacher" and "final_score" in df.columns:
        st.markdown("---")
        st.subheader("📈 Score Distribution")
        bins = [0, 40, 55, 70, 85, 100]
        labels = ["F (<40)", "D (40-55)", "C (55-70)", "B (70-85)", "A (85-100)"]
        df["grade_band"] = pd.cut(df["final_score"], bins=bins, labels=labels, right=True)
        grade_counts = df["grade_band"].value_counts().reindex(labels, fill_value=0)
        st.bar_chart(grade_counts)


# ─────────────────────────────────────────────────────────────
# PAGE 7: TEACHER DASHBOARD
# ─────────────────────────────────────────────────────────────

def page_teacher():
    st.markdown("## 👨‍🏫 Teacher Dashboard")

    all_results = get_all_results()
    if not all_results:
        st.info("No student submissions yet.")
        return

    df = pd.DataFrame(all_results)

    # ── OVERVIEW METRICS ──
    st.subheader("📊 Class Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_sub   = len(df)
    avg_score   = df["final_score"].mean() if "final_score" in df.columns else 0
    pass_count  = (df["final_score"] >= 55).sum() if "final_score" in df.columns else 0
    fail_count  = total_sub - pass_count
    unique_stu  = df["student"].nunique() if "student" in df.columns else 0

    col1.metric("👥 Students",          unique_stu)
    col2.metric("📝 Total Submissions", total_sub)
    col3.metric("📊 Class Average",     "{:.1f}%".format(avg_score))
    col4.metric("❌ Failed",             fail_count)

    st.markdown("---")

    # ── SUBJECT-WISE PERFORMANCE ──
    if "subject" in df.columns and "final_score" in df.columns:
        st.subheader("📚 Subject-wise Performance")
        subj_df = df.groupby("subject")["final_score"].agg(
            ["mean", "min", "max", "count"]
        ).round(2).reset_index()
        subj_df.columns = ["Subject", "Average", "Lowest", "Highest", "Submissions"]
        st.dataframe(subj_df, use_container_width=True)

        st.subheader("📈 Average Score by Subject")
        chart_data = subj_df.set_index("Subject")["Average"]
        st.bar_chart(chart_data)

    # ── STUDENTS NEEDING HELP ──
    st.markdown("---")
    st.subheader("⚠️ Students Who Need Re-Teaching")
    if "final_score" in df.columns and "student" in df.columns:
        struggling = df[df["final_score"] < 55][["student", "subject", "final_score", "grade"]]
        if len(struggling) > 0:
            st.dataframe(struggling.rename(columns={
                "student":     "Student",
                "subject":     "Subject",
                "final_score": "Score (%)",
                "grade":       "Grade",
            }), use_container_width=True)
        else:
            st.success("🎉 All students are performing above passing threshold!")

    # ── MISSING TOPICS ANALYSIS ──
    st.markdown("---")
    st.subheader("🔍 Commonly Missing Topics (Needs Re-Teaching)")
    if "missing_topics" in df.columns:
        all_missing = []
        for row in df["missing_topics"].dropna():
            if row:
                topics = [t.strip() for t in str(row).split(",") if t.strip()]
                all_missing.extend(topics)

        if all_missing:
            from collections import Counter
            topic_counts = Counter(all_missing).most_common(15)
            topic_df = pd.DataFrame(topic_counts, columns=["Topic", "Times Missing"])
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.bar_chart(topic_df.set_index("Topic")["Times Missing"])
            with col_b:
                st.dataframe(topic_df, use_container_width=True)
        else:
            st.info("No missing topic data available yet.")

    # ── STUDENT RANKING ──
    st.markdown("---")
    st.subheader("🏆 Student Ranking")
    if "student" in df.columns and "final_score" in df.columns:
        ranking = df.groupby("student")["final_score"].agg(["mean","count"]).round(2)
        ranking.columns = ["Average Score (%)", "Submissions"]
        ranking = ranking.sort_values("Average Score (%)", ascending=False).reset_index()
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
        ranking.rename(columns={"student": "Student"}, inplace=True)
        st.dataframe(ranking, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────

def main():
    if not st.session_state.logged_in:
        page_auth()
        return

    # Render sidebar after login
    render_sidebar()

    page = st.session_state.page

    if page == "dashboard":
        if st.session_state.role == "teacher":
            page_teacher()
        else:
            page_dashboard()

    elif page == "submit":
        if st.session_state.role == "student":
            page_submit()
        else:
            st.warning("Teachers cannot submit answers. Switch to student account.")

    elif page == "results":
        page_results()

    elif page == "feedback":
        if st.session_state.role == "student":
            page_feedback()
        else:
            page_results()

    elif page == "teacher":
        if st.session_state.role == "teacher":
            page_teacher()
        else:
            st.error("Access denied. Teacher only.")

    else:
        page_dashboard()


if __name__ == "__main__":
    main()
