import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- Page Config ---
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Load Data ---
@st.cache_data
def load_data():
    df_raw = pd.read_csv("C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Employee.csv")
    df_raw.dropna(inplace=True)
    return df_raw

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Model/Employee.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("Pipeline file not found.")
        return None

@st.cache_data
def get_model_features():
    df_processed = pd.read_csv("C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Preprocessed.csv")
    return df_processed.drop(columns=["Attrition_Yes"]).columns

# --- Load Assets ---
df = load_data()
pipeline = load_pipeline()
model_features = get_model_features()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Executive Dashboard", "Attrition Prediction"])

# --- Executive Dashboard ---
if section == "Executive Dashboard":
    st.title("Executive Dashboard: Attrition Insights")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Attrition Rate", f"{df['Attrition'].value_counts(normalize=True)['Yes']:.2%}")
    col3.metric("Avg Job Satisfaction", f"{df['JobSatisfaction'].mean():.2f} / 4")

    st.markdown("---")

    # Attrition by Department
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index()
        dept_attrition.columns = ["Department", "Attrition Rate"]
        fig = px.bar(dept_attrition, x="Department", y="Attrition Rate", text=dept_attrition["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Income vs. Attrition")
        fig = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition", notched=True)
        st.plotly_chart(fig, use_container_width=True)

    # Overtime & Job Satisfaction
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Rate by Overtime")
        over_attr = df.groupby("OverTime")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(over_attr, x="OverTime", y="Attrition Rate", text=over_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Job Satisfaction & Attrition")
        job_attr = df.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(job_attr, x="JobSatisfaction", y="Attrition Rate", text=job_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

    # Work-Life Balance & Role
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Work-Life Balance vs Attrition")
        wlb = df.groupby("WorkLifeBalance")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(wlb, x="WorkLifeBalance", y="Attrition Rate", text=wlb["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution by Job Role")
        fig = px.pie(df, names="JobRole", title="Job Role Distribution", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    # Age & Gender
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(df, x="Age", color="Attrition", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Gender-wise Attrition")
        gen_attr = df.groupby("Gender")["Attrition"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Attrition Rate")
        fig = px.bar(gen_attr, x="Gender", y="Attrition Rate", text=gen_attr["Attrition Rate"].apply(lambda x: f"{x:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

# --- Attrition Prediction ---
elif section == "Attrition Prediction":
    st.title("Attrition Prediction")

    if pipeline:
        st.markdown("Fill in the employee details to predict whether they are likely to leave.")

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
            # Get mode/median-based template for default values
            base = {}
            for col in df.columns:
                if col != "Attrition":
                    base[col] = df[col].mode()[0] if df[col].dtype == "object" else df[col].median()

            # Update with user inputs
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

            # Convert to DataFrame
            input_df = pd.DataFrame([base])

            # One-hot encode to match training
            input_encoded = pd.get_dummies(input_df)

            # Reindex to match training columns
            input_final = input_encoded.reindex(columns=model_features, fill_value=0)

            # Get prediction and probabilities
            prediction = pipeline.predict(input_final)[0]
            probas = pipeline.predict_proba(input_final)[0]

            # Extract individual class probabilities
            prob_stay = probas[0] * 100
            prob_leave = probas[1] * 100

            # Display Results
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(f"Prediction: Employee is **likely to leave**")
            else:
                st.success(f"Prediction: Employee is **likely to stay**")

            st.info(f"Confidence Scores:\n- Stay: **{prob_stay:.2f}%**\n- Leave: **{prob_leave:.2f}%**")

            # Show full input summary
            with st.expander("Input Summary"):
                st.json(base)
