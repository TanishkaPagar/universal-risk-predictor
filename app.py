import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import train_model
from predict import predict_single, predict_bulk
from utils import generate_excel_report

st.set_page_config(
    page_title="Employee Burnout & Attrition Risk Predictor",
    page_icon="👥",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👥 Employee Burnout & Attrition Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Universal HR Intelligence Tool — Works for any industry, any company</div>', unsafe_allow_html=True)
st.markdown("---")

# ===================== TRAIN MODEL =====================
@st.cache_resource
def get_model():
    import pandas as pd
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return train_model(df, target_col='Attrition')

with st.spinner("Loading AI model..."):
    model_data = get_model()

st.success(f"✅ Model Ready | Accuracy: {model_data['accuracy']*100:.1f}% | AUC: {model_data['auc']:.2f}")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Single Employee Prediction", "📂 Bulk Prediction (Upload CSV)"])

# ===================== TAB 1: SINGLE =====================
with tab1:
    st.subheader("Enter Employee Details")
    st.info("Fill in the details below for any employee — works for any industry!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.slider("Distance from Home (km)", 1, 100, 10)
        num_companies = st.slider("Number of Companies Worked At", 0, 20, 2)

    with col2:
        st.markdown("**💼 Job Info**")
        department = st.text_input("Department", placeholder="e.g. Sales, Nursing, Teaching")
        job_role = st.text_input("Job Role", placeholder="e.g. Manager, Doctor, Engineer")
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Entry",2:"Junior",3:"Mid",4:"Senior",5:"Executive"}[x])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        overtime = st.selectbox("Works Overtime?", ["Yes", "No"])

    with col3:
        st.markdown("**💰 Compensation & Experience**")
        monthly_income = st.number_input("Monthly Income (₹/$)", 1000, 500000, 50000, step=1000)
        percent_hike = st.slider("Salary Hike Last Year (%)", 0, 50, 10)
        total_working_years = st.slider("Total Working Years", 0, 45, 8)
        years_at_company = st.slider("Years at Current Company", 0, 45, 4)
        years_in_role = st.slider("Years in Current Role", 0, 25, 3)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 20, 2)

    st.markdown("**⭐ Satisfaction Ratings**")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    with r2:
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    with r3:
        env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    with r4:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    with r5:
        job_involvement = st.slider("Job Involvement", 1, 4, 3)

    training_times = st.slider("Training Sessions Last Year", 0, 10, 2)

    if st.button("🔮 Predict Burnout & Attrition Risk", type="primary", use_container_width=True):
        input_data = {
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'DistanceFromHome': distance,
            'Department': department if department else 'Unknown',
            'JobRole': job_role if job_role else 'Unknown',
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'OverTime': overtime,
            'MonthlyIncome': monthly_income,
            'PercentSalaryHike': percent_hike,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo,
            'WorkLifeBalance': work_life_balance,
            'JobInvolvement': job_involvement,
            'EnvironmentSatisfaction': env_satisfaction,
            'RelationshipSatisfaction': relationship_satisfaction,
            'NumCompaniesWorked': num_companies,
            'TrainingTimesLastYear': training_times,
            'BusinessTravel': business_travel
        }

        with st.spinner("Analyzing employee profile..."):
            prob, risk_label, factors = predict_single(input_data, model_data)

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            color = "#00C851" if "Low" in risk_label else "#FFD700" if "Medium" in risk_label else "#FF4444"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={'text': "Attrition Risk Score", 'font': {'size': 18}},
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
                        f'<div class="factor-card"><b>{f["feature"]}</b>: {f["value"]:.1f} — {direction} '
                        f'(impact: {f["impact"]:+.3f})</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("")
            if "High" in risk_label:
                st.error("⚠️ High Risk! Immediate retention action recommended.")
            elif "Medium" in risk_label:
                st.warning("👀 Medium Risk. Schedule a check-in with this employee.")
            else:
                st.success("✅ Low Risk. Employee appears stable and engaged.")

# ===================== TAB 2: BULK =====================
with tab2:
    st.subheader("Bulk Employee Risk Analysis")
    st.info("Upload a CSV with employee data. Columns should include: Age, Department, JobRole, JobSatisfaction, OverTime, MonthlyIncome, YearsAtCompany, WorkLifeBalance etc.")

    uploaded = st.file_uploader("📂 Upload Employee CSV", type=["csv"])

    if uploaded:
        df_bulk = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_bulk)} employees")
        st.dataframe(df_bulk.head(5), use_container_width=True)

        if st.button("🚀 Run Bulk Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing all employees..."):
                probs, labels = predict_bulk(df_bulk, model_data)

            df_result = df_bulk.copy()
            df_result['Risk Score (%)'] = (np.array(probs) * 100).round(1)
            df_result['Risk Level'] = labels

            st.markdown("---")
            low = sum(1 for l in labels if "Low" in l)
            med = sum(1 for l in labels if "Medium" in l)
            high = sum(1 for l in labels if "High" in l)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Employees", len(labels))
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

            excel_buf = generate_excel_report(df_bulk, probs, labels)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buf,
                file_name="burnout_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            