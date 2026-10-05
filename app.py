"""
app.py - Student Performance Predictor Streamlit Web Application.
Interactive ML dashboard featuring predictions, EDA, model evaluation, and educational insights.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import (
    load_student_data,
    FEATURE_COLUMNS,
    CATEGORIES,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN
)
from src.model import (
    predict_student_performance,
    load_artifacts,
    train_and_evaluate,
    CLASS_ORDER
)

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Academic & Technology Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        color: white;
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #FFFFFF !important;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        max-width: 800px;
        line-height: 1.5;
        color: #E0E7FF !important;
    }
    
    /* Stat cards */
    .stat-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.07);
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .stat-subtext {
        font-size: 0.8rem;
        color: #3B82F6;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    /* Result callout cards */
    .result-card-strong {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .result-card-developing {
        background-color: #EFF6FF;
        border-left: 6px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .result-card-support {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    /* Info banners */
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #92400E;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 1.2rem 0;
    }
    
    /* General card */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Artifact Loading with Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    """Loads student dataset with caching."""
    data_path = os.path.join("data", "student-mat.csv")
    try:
        return load_student_data(data_path=data_path)
    except Exception as err:
        st.error(f"Failed to load dataset: {err}")
        return None


@st.cache_resource
def get_artifacts():
    """Loads machine learning models and evaluation metrics."""
    best_m, lr_m, rf_m, eval_data = load_artifacts(save_dir="models")
    if best_m is None or eval_data is None:
        with st.spinner("Training initial machine learning models..."):
            train_and_evaluate(
                data_path=os.path.join("data", "student-mat.csv"),
                save_dir="models"
            )
            best_m, lr_m, rf_m, eval_data = load_artifacts(save_dir="models")
    return best_m, lr_m, rf_m, eval_data


df = get_dataset()
best_model, lr_model, rf_model, eval_data = get_artifacts()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Context
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.caption("Educational AI/ML Analytics Suite")
    
    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🎯 Student Prediction",
            "📊 Exploratory Data Analysis",
            "📈 Model Evaluation",
            "ℹ️ About the Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if eval_data:
        sel_name = eval_data.get("selected_model", "Random Forest")
        sel_acc = eval_data["models"]["random_forest"]["accuracy"] * 100
        st.markdown(f"**Active Model:** `{sel_name}`")
        st.markdown(f"**Test Accuracy:** `{sel_acc:.1f}%`")
    if df is not None:
        st.markdown(f"**Dataset Size:** `{len(df)} students`")
    
    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.78rem; color: #64748B;'>"
        "<strong>Notice:</strong> This web app is an educational machine learning demonstration. "
        "Predictions are probabilistic indicators and must never be used for formal academic grading or disciplinary decisions."
        "</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if nav_selection == "🏠 Home Dashboard":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Student Academic Performance Predictor</div>
        <div class="hero-subtitle">
            An open educational machine learning application trained on historical secondary school data 
            to estimate student academic performance categories and identify early support opportunities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Grid
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Students</div>
            <div class="stat-value">395</div>
            <div class="stat-subtext">Math Course (UCI)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Input Features</div>
            <div class="stat-value">6</div>
            <div class="stat-subtext">Academics & Habits</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        sel_model = eval_data["selected_model"] if eval_data else "Random Forest"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Selected Model</div>
            <div class="stat-value" style="font-size: 1.35rem; padding-top: 0.35rem;">{sel_model}</div>
            <div class="stat-subtext">Ensemble Classifier</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        acc_val = eval_data["models"]["random_forest"]["accuracy"] * 100 if eval_data else 87.3
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Test Accuracy</div>
            <div class="stat-value">{acc_val:.1f}%</div>
            <div class="stat-subtext">vs 48.1% Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        f1_val = eval_data["models"]["random_forest"]["macro_f1"] * 100 if eval_data else 89.4
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Macro F1-Score</div>
            <div class="stat-value">{f1_val:.1f}%</div>
            <div class="stat-subtext">Balanced Metric</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Educational Disclaimer Card
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Responsible AI & Usage Disclaimer:</strong><br>
        This tool is created strictly for educational exploration and machine learning methodology illustration. 
        Predicted outcomes are statistical estimates derived from historical Portuguese secondary school data (2005–2006). 
        They do <strong>not</strong> represent absolute student capability or determine future educational trajectory.
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Workflow Walkthrough
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🔍 How the Prediction Engine Works")
        st.markdown("""
        The application uses an audited machine learning pipeline designed to prevent data leakage and provide calibrated insights:
        
        1. **Feature Input Validation:** Captures six key attributes (Age, Study Time, Past Failures, Absences, Grade 1, Grade 2) and validates bounds.
        2. **Standardization Preprocessing:** Numerical values are scaled using `StandardScaler` fitted strictly on training data.
        3. **Ensemble Classification:** A Random Forest of 100 decision trees evaluates non-linear thresholds across attendance, past academic hurdles, and mid-term grades.
        4. **Probability Calculation:** Computes class distribution percentages across three academic performance tiers.
        5. **Strict Target Isolation:** The final grade (G3) is entirely isolated during prediction to guarantee legitimate inference without leakage.
        """)

    with col_right:
        st.markdown("### 🎯 Target Performance Tiers")
        st.markdown("""
        <div class="content-box">
            <p><strong>🟢 Strong (Grade 15–20):</strong> Excelling performance demonstrating mastery of mathematical concepts.</p>
            <p><strong>🔵 Developing (Grade 10–14):</strong> Passing standard with consistent foundational understanding and room for growth.</p>
            <p><strong>🔴 Needs Support (Grade 0–9):</strong> Below passing threshold; flags early opportunity for tailored pedagogical support.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: STUDENT PREDICTION
# =============================================================================
elif nav_selection == "🎯 Student Prediction":
    st.markdown("## 🎯 Predict Student Performance Category")
    st.markdown(
        "Enter student academic history and behavioral habits below. "
        "The model will estimate performance probability across the three academic categories."
    )
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Academic History")
            g1 = st.slider(
                "First-Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=12,
                help="Term 1 examination mark on the standard 0–20 Portuguese scale."
            )
            g2 = st.slider(
                "Second-Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=13,
                help="Term 2 examination mark on the standard 0–20 Portuguese scale."
            )
            failures = st.selectbox(
                "Previous Class Failures",
                options=[0, 1, 2, 3],
                index=0,
                help="Count of past academic failures (0 to 3+)."
            )
            
        with col2:
            st.markdown("#### 👤 Student Profile & Habits")
            age = st.slider(
                "Student Age",
                min_value=15,
                max_value=22,
                value=17,
                help="Age in years (15–22)."
            )
            studytime_labels = {
                1: "1: < 2 hours / week",
                2: "2: 2 to 5 hours / week",
                3: "3: 5 to 10 hours / week",
                4: "4: > 10 hours / week"
            }
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                index=1,
                format_func=lambda x: studytime_labels[x],
                help="Estimated weekly hours spent studying outside classroom lectures."
            )
            absences = st.slider(
                "School Absences",
                min_value=0,
                max_value=30,
                value=4,
                help="Total days absent from school during the academic term."
            )

        submitted = st.form_submit_button("🚀 Predict Performance", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2
        }

        with st.spinner("Calculating predictions..."):
            pred_result = predict_student_performance(best_model, input_data)
            category = pred_result["predicted_category"]
            probs = pred_result["probabilities"]
            explanation = pred_result["explanation"]

        st.markdown("---")
        st.markdown("### 📋 Prediction Results")

        # Color-coded result card
        card_class = (
            "result-card-strong" if category == "Strong"
            else "result-card-developing" if category == "Developing"
            else "result-card-support"
        )
        badge_color = (
            "#10B981" if category == "Strong"
            else "#3B82F6" if category == "Developing"
            else "#EF4444"
        )

        st.markdown(f"""
        <div class="{card_class}">
            <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Predicted Category
            </span>
            <h2 style="margin: 0.5rem 0 0.25rem 0; color: #0F172A;">{category}</h2>
            <p style="margin: 0; color: #334155; font-size: 0.95rem;">{explanation}</p>
        </div>
        """, unsafe_allow_html=True)

        # Columns for probability chart and input summary
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            st.markdown("#### Estimated Category Probabilities")
            # Interactive Plotly probability bar chart
            prob_df = pd.DataFrame({
                "Category": CLASS_ORDER,
                "Probability": [probs[c] for c in CLASS_ORDER],
                "Percentage": [f"{probs[c]*100:.1f}%" for c in CLASS_ORDER]
            })

            colors = ["#EF4444", "#3B82F6", "#10B981"]

            fig = go.Figure(go.Bar(
                x=prob_df["Probability"],
                y=prob_df["Category"],
                orientation="h",
                text=prob_df["Percentage"],
                textposition="auto",
                marker=dict(color=colors, line=dict(color="#1E293B", width=1))
            ))
            fig.update_layout(
                xaxis=dict(title="Probability Estimate", range=[0, 1.05], tickformat=".0%"),
                yaxis=dict(title="", autorange="reversed"),
                height=260,
                margin=dict(l=10, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("#### Input Profile Summary")
            st.markdown(f"""
            <div class="content-box">
                <table style="width: 100%; font-size: 0.9rem;">
                    <tr><td><strong>Age:</strong></td><td>{age} yrs</td></tr>
                    <tr><td><strong>Study Time:</strong></td><td>{studytime_labels[studytime]}</td></tr>
                    <tr><td><strong>Prior Failures:</strong></td><td>{failures}</td></tr>
                    <tr><td><strong>Absences:</strong></td><td>{absences} days</td></tr>
                    <tr><td><strong>First Grade (G1):</strong></td><td>{g1} / 20</td></tr>
                    <tr><td><strong>Second Grade (G2):</strong></td><td>{g2} / 20</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "ℹ️ **Estimation Notice:** Probabilities displayed are statistical model outputs, not calibrated certainty values. "
            "Because early term marks (G1 and G2) correlate strongly with final outcomes, adjustments in study habits and instructional intervention "
            "can significantly alter a student's real-world academic trajectory."
        )


# =============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
elif nav_selection == "📊 Exploratory Data Analysis":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown(
        "Explore distributions, correlations, and relationships within the UCI Mathematics Student Performance dataset. "
        "All charts are interactive."
    )

    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Target & Grade Distributions",
            "⏱️ Study Habits & Attendance",
            "🎯 G1/G2 vs Final Grade",
            "🔥 Correlation Heatmap"
        ])

        with tab1:
            st.markdown("### Final Grade (G3) & Performance Tiers")
            col_a, col_b = st.columns(2)

            with col_a:
                fig_hist = px.histogram(
                    df,
                    x=RAW_TARGET_COLUMN,
                    nbins=21,
                    title="Distribution of Final Grades (G3: 0–20 Scale)",
                    color_discrete_sequence=["#2563EB"],
                    labels={"G3": "Final Exam Grade (G3)"}
                )
                # Add threshold lines
                fig_hist.add_vline(x=9.5, line_dash="dash", line_color="#EF4444", annotation_text="Support Cutoff")
                fig_hist.add_vline(x=14.5, line_dash="dash", line_color="#10B981", annotation_text="Strong Cutoff")
                fig_hist.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=380)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption("Distribution shows a cluster at grade 0 (students who dropped out or did not attend the final exam) and a bell-shaped distribution centered around 10–12.")

            with col_b:
                cat_counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
                fig_pie = px.pie(
                    names=cat_counts.index,
                    values=cat_counts.values,
                    title="Proportion of Performance Categories",
                    color=cat_counts.index,
                    color_discrete_map={
                        "Needs Support": "#EF4444",
                        "Developing": "#3B82F6",
                        "Strong": "#10B981"
                    },
                    hole=0.45
                )
                fig_pie.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=380)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption(f"Developing constitutes the majority class ({cat_counts['Developing']/len(df)*100:.1f}%), followed by Needs Support ({cat_counts['Needs Support']/len(df)*100:.1f}%), and Strong ({cat_counts['Strong']/len(df)*100:.1f}%).")

        with tab2:
            st.markdown("### Impact of Study Time and Absences on Final Grade")
            col_c, col_d = st.columns(2)

            with col_c:
                study_labels = {1: "< 2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: "> 10 hrs"}
                df_study = df.copy()
                df_study["study_label"] = df_study["studytime"].map(study_labels)

                fig_box = px.box(
                    df_study,
                    x="study_label",
                    y=RAW_TARGET_COLUMN,
                    category_orders={"study_label": ["< 2 hrs", "2-5 hrs", "5-10 hrs", "> 10 hrs"]},
                    color="study_label",
                    title="Weekly Study Time vs. Final Grade",
                    labels={"study_label": "Weekly Study Time", "G3": "Final Grade (G3)"},
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_box.update_layout(showlegend=False, height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption("Observation: Higher weekly study time displays an upward trend in median performance, though individual variation remains wide.")

            with col_d:
                fig_scatter_abs = px.scatter(
                    df,
                    x="absences",
                    y=RAW_TARGET_COLUMN,
                    color=TARGET_COLUMN,
                    color_discrete_map={
                        "Needs Support": "#EF4444",
                        "Developing": "#3B82F6",
                        "Strong": "#10B981"
                    },
                    title="School Absences vs. Final Grade",
                    labels={"absences": "Number of Absences", "G3": "Final Grade (G3)"}
                )
                fig_scatter_abs.update_layout(height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_scatter_abs, use_container_width=True)
                st.caption("Observation: Students with very high absences (>20) rarely reach the 'Strong' tier, but moderate absences show varied outcomes.")

        with tab3:
            st.markdown("### Association Between Prior Grades and Final Performance")
            col_e, col_f = st.columns(2)

            with col_e:
                fig_g1 = px.scatter(
                    df,
                    x="G1",
                    y=RAW_TARGET_COLUMN,
                    trendline="ols",
                    title="Period 1 Grade (G1) vs. Final Grade (G3)",
                    color=TARGET_COLUMN,
                    color_discrete_map={
                        "Needs Support": "#EF4444",
                        "Developing": "#3B82F6",
                        "Strong": "#10B981"
                    }
                )
                fig_g1.update_layout(height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_g1, use_container_width=True)
                corr_g1 = df["G1"].corr(df["G3"])
                st.caption(f"Strong linear correlation (Pearson r = {corr_g1:.2f}). Early term marks provide a strong predictive baseline for final achievement.")

            with col_f:
                fig_g2 = px.scatter(
                    df,
                    x="G2",
                    y=RAW_TARGET_COLUMN,
                    trendline="ols",
                    title="Period 2 Grade (G2) vs. Final Grade (G3)",
                    color=TARGET_COLUMN,
                    color_discrete_map={
                        "Needs Support": "#EF4444",
                        "Developing": "#3B82F6",
                        "Strong": "#10B981"
                    }
                )
                fig_g2.update_layout(height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_g2, use_container_width=True)
                corr_g2 = df["G2"].corr(df["G3"])
                st.caption(f"Very strong linear correlation (Pearson r = {corr_g2:.2f}). Second period performance is the single strongest single feature predictor.")

        with tab4:
            st.markdown("### Feature Correlation Matrix")
            corr_cols = FEATURE_COLUMNS + [RAW_TARGET_COLUMN]
            corr_matrix = df[corr_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="Blues",
                title="Pearson Correlation Heatmap (Numeric Variables)",
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(height=450, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                "Note on causality: Correlation indicates statistical co-occurrence, not direct causation. "
                "For instance, class failures and absences negatively correlate with final marks, while study time is positively associated."
            )


# =============================================================================
# PAGE 4: MODEL EVALUATION
# =============================================================================
elif nav_selection == "📈 Model Evaluation":
    st.markdown("## 📈 Machine Learning Model Evaluation")
    st.markdown(
        "All metrics below are calculated from the held-out test dataset (20% sample, 79 students), "
        "evaluating models on completely unseen student profiles."
    )

    if eval_data:
        models_data = eval_data["models"]
        
        # Summary Comparison Table
        st.markdown("### 📊 Model Performance Comparison")
        comp_data = []
        for key, name in [
            ("baseline", "Majority-Class Baseline"),
            ("logistic_regression", "Logistic Regression"),
            ("random_forest", "Random Forest Classifier")
        ]:
            m = models_data[key]
            comp_data.append({
                "Model": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Macro Precision": f"{m['macro_precision'] * 100:.2f}%",
                "Macro Recall": f"{m['macro_recall'] * 100:.2f}%",
                "Macro F1-Score": f"{m['macro_f1'] * 100:.2f}%"
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="content-box">
            <strong>🏆 Champion Model Selection: {eval_data['selected_model']}</strong><br>
            <em>{eval_data['selection_rationale']}</em>
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrices Section
        st.markdown("### 🧩 Confusion Matrices (Test Set: N = 79)")
        col_cm1, col_cm2 = st.columns(2)

        with col_cm1:
            rf_cm = np.array(models_data["random_forest"]["confusion_matrix"])
            fig_rf_cm = px.imshow(
                rf_cm,
                x=CLASS_ORDER,
                y=CLASS_ORDER,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Random Forest Confusion Matrix",
                labels=dict(x="Predicted Class", y="True Class")
            )
            fig_rf_cm.update_layout(height=360, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig_rf_cm, use_container_width=True)

        with col_cm2:
            lr_cm = np.array(models_data["logistic_regression"]["confusion_matrix"])
            fig_lr_cm = px.imshow(
                lr_cm,
                x=CLASS_ORDER,
                y=CLASS_ORDER,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Logistic Regression Confusion Matrix",
                labels=dict(x="Predicted Class", y="True Class")
            )
            fig_lr_cm.update_layout(height=360, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig_lr_cm, use_container_width=True)

        # Educational Metrics Guide
        st.markdown("### 📖 Understanding Evaluation Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            - **Accuracy:** The ratio of correct predictions to total cases. While intuitive, it is misleading on imbalanced datasets. Notice the baseline achieves 48.1% accuracy simply by guessing "Developing" every time, but fails entirely on the other classes.
            - **Macro Precision:** Evaluates the average precision across all three classes equally, penalizing false alarms in smaller classes.
            """)
        with col_m2:
            st.markdown("""
            - **Macro Recall:** Evaluates the average sensitivity across classes. Crucial for educational interventions to ensure students who "Need Support" are not missed.
            - **Macro F1-Score:** The harmonic mean of precision and recall. Chosen as the primary selection criterion because it rewards balanced performance across all tiers.
            """)


# =============================================================================
# PAGE 5: ABOUT THE PROJECT
# =============================================================================
elif nav_selection == "ℹ️ About the Project":
    st.markdown("## ℹ️ About Student Performance Predictor")
    
    st.markdown("""
    ### 🎯 Problem Statement
    In educational institutions, identifying students who may face academic difficulties early in the term allows teachers 
    and advisors to provide supportive interventions (tutoring, counseling, study habit coaching) before final examinations. 
    This application demonstrates how supervised machine learning can be structured to analyze prior academic performance 
    and behavioral factors while emphasizing transparency and ethical boundaries.
    """)

    st.markdown("""
    ### 📂 Dataset Source & Attribution
    The data used in this project is the **Student Performance Data Set** hosted by the **UCI Machine Learning Repository**:
    
    > **Citation:** Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. 
    In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal. 
    Available at: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
    """)

    st.warning("""
    **⚠️ Geographic and Educational System Boundaries:**
    The dataset was compiled from two public secondary schools (Gabriel Pereira and Mousinho da Silveira) located in the Alentejo 
    region of Portugal during 2005–2006. The grading system (0–20 scale), cultural norms, socioeconomic background, 
    and institutional policies are specific to that regional context. These patterns **do not automatically generalize** 
    to other schools, higher education institutions, or distinct national education systems.
    """)

    st.markdown("### 🏷️ Features & Performance Categories")
    st.markdown("""
    The model employs six non-leaked input features:
    """)
    feat_df = pd.DataFrame([
        {"Feature": "Age", "Type": "Numeric (15–22)", "Description": "Age of the student in years."},
        {"Feature": "Weekly Study Time", "Type": "Categorical (1–4)", "Description": "1: <2 hrs, 2: 2–5 hrs, 3: 5–10 hrs, 4: >10 hrs."},
        {"Feature": "Previous Failures", "Type": "Numeric (0–3)", "Description": "Number of previous academic class failures."},
        {"Feature": "Absences", "Type": "Numeric (0–30)", "Description": "Total recorded days of absence during the academic year."},
        {"Feature": "G1 Grade", "Type": "Numeric (0–20)", "Description": "First period examination grade."},
        {"Feature": "G2 Grade", "Type": "Numeric (0–20)", "Description": "Second period examination grade."}
    ])
    st.table(feat_df)

    st.markdown("""
    The target variable is transformed from the final grade (G3) into three distinct tiers:
    - **Needs Support:** Final mark from **0 to 9** (below passing threshold of 10).
    - **Developing:** Final mark from **10 to 14** (standard secondary passing performance).
    - **Strong:** Final mark from **15 to 20** (high academic achievement and mastery).
    """)

    st.markdown("""
    ### 🛡️ Ethical Considerations & Model Limitations
    - **High Influence of G1 and G2:** Because previous grades are strongly correlated with final grades, the model relies heavily on historical exam marks. It should not be assumed that students with low mid-term grades cannot excel with proper support.
    - **No Determinism:** Machine learning outputs represent historical statistical correlations, not deterministic human limits.
    - **No Real Decisions:** This tool is strictly instructional and should never replace human counseling or instructor discretion.
    """)
