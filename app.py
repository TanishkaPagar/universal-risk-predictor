import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import detect_target_column, train_on_any_dataset
from predict import predict_bulk, predict_single, get_risk_label
from utils import generate_excel_report

st.set_page_config(
    page_title="Universal Risk Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F3864;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .low { background-color: #d4edda; color: #155724; }
    .medium { background-color: #fff3cd; color: #856404; }
    .high { background-color: #f8d7da; color: #721c24; }
    .factor-card {
        background: #f8f9fa;
        border-left: 4px solid #1F3864;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #000000;
    }
    .info-box {
        background: #e8f4fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">🤖 Universal Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload ANY dataset → Auto-trains ML model → Predicts risk for every row</div>', unsafe_allow_html=True)
st.markdown("---")

# ===================== STEP 1: UPLOAD & TRAIN =====================
st.markdown("## Step 1 — Upload Your Dataset & Train Model")
st.info("Upload any CSV dataset. The app will auto-detect the target column, preprocess data, and train an XGBoost model.")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded — {df.shape[0]} rows, {df.shape[1]} columns")

    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head(10), use_container_width=True)

    # Target column selection
    suggested_target = detect_target_column(df)
    target_col = st.selectbox(
        "🎯 Select Target Column (what to predict)",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(suggested_target)
    )

    st.write(f"**Target column:** `{target_col}` | **Unique values:** {df[target_col].nunique()} → {df[target_col].unique()[:5]}")

    if st.button("🚀 Train Model on This Dataset", type="primary", use_container_width=True):
        with st.spinner("Auto-preprocessing and training model... please wait"):
            try:
                model_data = train_on_any_dataset(df, target_col)
                st.session_state['model_data'] = model_data
                st.session_state['df'] = df
                st.session_state['target_col'] = target_col
                st.success("✅ Model trained successfully!")

                # Show metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{model_data['accuracy']*100:.1f}%")
                m2.metric("ROC-AUC Score", f"{model_data['auc']:.4f}")
                m3.metric("Features Used", len(model_data['features']))

                st.info(f"🎯 Predicting: **{target_col}** | Classes: {model_data['classes']}")

            except Exception as e:
                st.error(f"Error during training: {str(e)}")

# ===================== STEP 2: PREDICT =====================
if 'model_data' in st.session_state:
    model_data = st.session_state['model_data']
    df = st.session_state['df']
    target_col = st.session_state['target_col']

    st.markdown("---")
    st.markdown("## Step 2 — Predict Risk")

    tab1, tab2 = st.tabs(["📂 Bulk Prediction (Full Dataset)", "🔍 Single Row Prediction"])

    # ===== TAB 1: BULK =====
    with tab1:
        st.subheader("Predict Risk for All Rows")

        if st.button("🔮 Run Bulk Prediction", type="primary", use_container_width=True):
            with st.spinner("Scoring all rows..."):
                probs, labels = predict_bulk(df, model_data)

            df_result = df.copy()
            df_result['Risk Score (%)'] = (np.array(probs) * 100).round(1)
            df_result['Risk Level'] = labels

            st.markdown("---")
            low = sum(1 for l in labels if "Low" in l)
            med = sum(1 for l in labels if "Medium" in l)
            high = sum(1 for l in labels if "High" in l)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Rows", len(labels))
            m2.metric("🟢 Low Risk", low)
            m3.metric("🟡 Medium Risk", med)
            m4.metric("🔴 High Risk", high)

            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    values=[low, med, high],
                    names=["Low Risk", "Medium Risk", "High Risk"],
                    color_discrete_sequence=["#00C851", "#FFD700", "#FF4444"],
                    title="Risk Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_hist = px.histogram(
                    df_result,
                    x='Risk Score (%)',
                    nbins=20,
                    title="Risk Score Distribution",
                    color_discrete_sequence=["#1F3864"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### 📋 Full Results")
            st.dataframe(
                df_result.sort_values('Risk Score (%)', ascending=False),
                use_container_width=True
            )

            excel_buf = generate_excel_report(df, probs, labels)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buf,
                file_name="risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # ===== TAB 2: SINGLE =====
    with tab2:
        st.subheader("Predict Risk for a Single Entry")
        st.info("Fill in values for each feature below")

        features = model_data['features']
        input_dict = {}
        original_df = df.drop(columns=[target_col], errors='ignore')

        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                if feature in original_df.columns:
                    col_data = original_df[feature]
                    if col_data.dtype == object:
                        options = col_data.dropna().unique().tolist()
                        input_dict[feature] = st.selectbox(f"{feature}", options)
                    else:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        mean_val = float(col_data.mean())
                        input_dict[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val
                        )
                else:
                    input_dict[feature] = st.number_input(f"{feature}", value=0.0)

        if st.button("🔮 Predict This Entry", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                prob, risk_label, factors = predict_single(input_dict, model_data)

            st.markdown("---")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                color = "#00C851" if "Low" in risk_label else "#FFD700" if "Medium" in risk_label else "#FF4444"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    title={'text': "Risk Score", 'font': {'size': 18}},
                    number={'suffix': "%", 'font': {'size': 36}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 60], 'color': '#fff3cd'},
                            {'range': [60, 100], 'color': '#f8d7da'},
                        ],
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

                risk_class = "low" if "Low" in risk_label else "medium" if "Medium" in risk_label else "high"
                st.markdown(f'<div class="risk-card {risk_class}">{risk_label}</div>', unsafe_allow_html=True)

            with col_res2:
                st.markdown("#### 🔍 Top Factors Driving This Prediction")
                st.markdown("*(SHAP-based explainability)*")
                if factors:
                    for f in factors:
                        direction = "⬆️ Increases risk" if f['impact'] > 0 else "⬇️ Decreases risk"
                        st.markdown(
                            f'<div class="factor-card"><b>{f["feature"]}</b>: {f["value"]:.2f} — {direction} '
                            f'(impact: {f["impact"]:+.3f})</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("SHAP factors not available for this prediction.")

                if "High" in risk_label:
                    st.error("⚠️ High risk detected! Immediate attention recommended.")
                elif "Medium" in risk_label:
                    st.warning("👀 Medium risk. Monitor closely.")
                else:
                    st.success("✅ Low risk. Looking stable!")