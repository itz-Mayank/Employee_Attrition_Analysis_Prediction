import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide",
)

# File Paths
BASE_PATH = r"C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/new_model"
RAW_FILE = os.path.join(BASE_PATH, "Employee.csv")
MODEL_FILE = os.path.join(BASE_PATH, "Employee.pkl")
SCALER_FILE = os.path.join(BASE_PATH, "scaler.pkl")
FEATURES_FILE = os.path.join(BASE_PATH, "features.pkl")

# Load Data
@st.cache_data
def load_data():
    df_raw = pd.read_csv(RAW_FILE)
    df_raw.dropna(inplace=True)
    return df_raw

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please retrain.")
        return None, None

@st.cache_data
def load_features():
    try:
        features = joblib.load(FEATURES_FILE)
        return features
    except Exception as e:
        st.error(f"Error loading feature columns: {e}")
        return []

# Load Assets
df = load_data()
model, scaler = load_model_and_scaler()
model_features = load_features()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Executive Dashboard", "Attrition Prediction"])


# Executive Dashboard
if section == "Executive Dashboard":
    st.title("Executive Dashboard: Attrition Insights")

    # Top KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Attrition Rate", f"{df['Attrition'].value_counts(normalize=True)['Yes']:.2%}")
    col3.metric("Avg Job Satisfaction", f"{df['JobSatisfaction'].mean():.2f} / 4")

    st.markdown("---")

    # Attrition by Department
    col1, col2 = st.columns(2)
    with col1:
        dept_attrition = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index()
        dept_attrition.columns = ["Department", "Attrition Rate"]
        fig = px.bar(dept_attrition, x="Department", y="Attrition Rate",
                     text=dept_attrition["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.subheader("Attrition by Department")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition", notched=True)
        st.subheader("Monthly Income vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    # Overtime & Job Satisfaction
    col1, col2 = st.columns(2)
    with col1:
        over_attr = df.groupby("OverTime")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(over_attr, x="OverTime", y="Attrition Rate",
                     text=over_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.subheader("⏱ Attrition Rate by Overtime")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        job_attr = df.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(job_attr, x="JobSatisfaction", y="Attrition Rate",
                     text=job_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.subheader("Job Satisfaction vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    # Work-Life Balance & Job Role
    col1, col2 = st.columns(2)
    with col1:
        wlb = df.groupby("WorkLifeBalance")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(wlb, x="WorkLifeBalance", y="Attrition Rate",
                     text=wlb["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.subheader("Work-Life Balance vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names="JobRole", title="Job Role Distribution", hole=0.3)
        st.subheader("Job Role Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Age & Gender
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age", color="Attrition", marginal="box")
        st.subheader("Age Distribution by Attrition")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gen_attr = df.groupby("Gender")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(gen_attr, x="Gender", y="Attrition Rate",
                     text=gen_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.subheader("Gender-wise Attrition")
        st.plotly_chart(fig, use_container_width=True)


# Attrition Prediction
elif section == "Attrition Prediction":
    st.title("Attrition Prediction")
    
    if model is not None and scaler is not None and len(model_features) > 0:
        st.markdown("### Fill in employee details to predict whether they are likely to leave.")

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 18, 70, 35)
                overtime = st.selectbox("OverTime", df["OverTime"].unique())
                marital = st.selectbox("Marital Status", df["MaritalStatus"].unique())

            with col2:
                income = st.number_input("Monthly Income", 1000, 20000, 5000)
                job_role = st.selectbox("Job Role", df["JobRole"].unique())
                dept = st.selectbox("Department", df["Department"].unique())

            with col3:
                job_sat = st.slider("Job Satisfaction", 1, 4, 3)
                total_years = st.slider("Total Working Years", 0, 40, 10)
                years_company = st.slider("Years at Company", 0, 40, 5)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # Default values using mode/median
            base = {}
            for col in df.columns:
                if col != "Attrition":
                    base[col] = df[col].mode()[0] if df[col].dtype == "object" else df[col].median()

            # Update with user input
            base.update({
                "Age": age,
                "OverTime": overtime,
                "MaritalStatus": marital,
                "MonthlyIncome": income,
                "JobRole": job_role,
                "Department": dept,
                "JobSatisfaction": job_sat,
                "TotalWorkingYears": total_years,
                "YearsAtCompany": years_company,
            })

            input_df = pd.DataFrame([base])
            input_encoded = pd.get_dummies(input_df)
            input_final = input_encoded.reindex(columns=model_features, fill_value=0)

            # Apply Scaler
            input_scaled = scaler.transform(input_final)

            # Prediction
            probas = model.predict_proba(input_scaled)[0]
            prediction = np.argmax(probas)

            prob_stay, prob_leave = probas[0] * 100, probas[1] * 100

            # Results
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Employee is **likely to leave**")
            else:
                st.success("Employee is **likely to stay**")

            # Input Summary
            with st.expander("Input Summary"):
                st.json(base)

    else:
        st.warning("Model, scaler, or features not loaded. Please retrain or check file paths.")
