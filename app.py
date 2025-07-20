import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Data and Models ---
@st.cache_data
def load_visualization_data():
    """Loads the raw data for creating charts."""
    df = pd.read_csv('C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Employee.csv')
    df.dropna(inplace=True)
    return df

@st.cache_data
def load_model_feature_list():
    """Loads the preprocessed data to get the column names the model expects."""
    df = pd.read_csv('C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Preprocessed.csv')
    df.dropna(inplace=True)
    model_features = df.drop(columns=['Attrition_Yes']).columns
    return model_features

@st.cache_resource
def load_models():
    """Loads the trained model and scaler."""
    try:
        attrition_model = joblib.load('C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Model/Employee.pkl')
        scaler = joblib.load('C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Model/scaler.pkl')
        return attrition_model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'Employee.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

# Load all components
df_viz = load_visualization_data()
model_features = load_model_feature_list()
attrition_model, scaler = load_models()

# --- Sidebar Navigation ---
st.sidebar.title("HR Analytics Navigation")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Attrition Prediction"])

# --- Executive Dashboard Page ---
if page == "Executive Dashboard":
    st.title("Executive Dashboard: Attrition Insights")
    st.markdown("### Key Company-Wide Metrics")

    # --- Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df_viz))
    col2.metric("Attrition Rate", f"{df_viz['Attrition'].value_counts(normalize=True)['Yes']:.2%}")
    col3.metric("Average Job Satisfaction", f"{df_viz['JobSatisfaction'].mean():.2f}/4")

    st.markdown("---")
    st.markdown("### Organizational & Financial Factors")

    # --- Row 1: Department and Income ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        attrition_rate_dept = df_viz.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='Attrition Rate')
        fig = px.bar(attrition_rate_dept, x='Department', y='Attrition Rate',
                     text=attrition_rate_dept['Attrition Rate'].apply(lambda x: f'{x:.2%}'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Income vs. Attrition")
        fig = px.box(df_viz, x='Attrition', y='MonthlyIncome', color='Attrition', notched=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Job & Work-Life Factors")

    # --- Row 2: Overtime, Job Satisfaction, and Work-Life Balance ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Rate by Overtime")
        attrition_rate_overtime = df_viz.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='Attrition Rate')
        fig = px.bar(attrition_rate_overtime, x='OverTime', y='Attrition Rate',
                     text=attrition_rate_overtime['Attrition Rate'].apply(lambda x: f'{x:.2%}'),
                     color='OverTime', color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Attrition Rate by Job Satisfaction")
        attrition_rate_satisfaction = df_viz.groupby('JobSatisfaction')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='Attrition Rate')
        fig = px.bar(attrition_rate_satisfaction, x='JobSatisfaction', y='Attrition Rate',
                     title="Attrition Rate by Job Satisfaction Level",
                     text=attrition_rate_satisfaction['Attrition Rate'].apply(lambda x: f'{x:.2%}'))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Work-Life Balance")
        attrition_rate_wlb = df_viz.groupby('WorkLifeBalance')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='Attrition Rate')
        fig = px.bar(attrition_rate_wlb, x='WorkLifeBalance', y='Attrition Rate',
                     title="Attrition Rate by Work-Life Balance Score",
                     text=attrition_rate_wlb['Attrition Rate'].apply(lambda x: f'{x:.2%}'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
         st.subheader("Distribution by Job Role")
         fig = px.pie(df_viz, names='JobRole', title='Employee Distribution by Job Role', hole=.3)
         st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")
    st.markdown("### Demographic Factors")

    # --- Row 3: Age and Gender ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution by Attrition Status")
        fig = px.histogram(df_viz, x='Age', color='Attrition', marginal='box',
                           title="Age Distribution of Employees")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Attrition by Gender")
        attrition_rate_gender = df_viz.groupby('Gender')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index(name='Attrition Rate')
        fig = px.bar(attrition_rate_gender, x='Gender', y='Attrition Rate',
                     text=attrition_rate_gender['Attrition Rate'].apply(lambda x: f'{x:.2%}'),
                     color='Gender', color_discrete_map={'Male':'#636EFA', 'Female':'#EF553B'})
        st.plotly_chart(fig, use_container_width=True)


# --- Attrition Prediction Page ---
elif page == "Attrition Prediction":
    st.title("Employee Attrition Prediction")
    st.markdown("### Predict the likelihood of an employee leaving the company.")

    if attrition_model and scaler and model_features is not None:
        with st.form("attrition_form"):
            st.subheader("Enter Employee Details to Predict Attrition")

            # Use the raw visualization data (df_viz) for user-friendly dropdown options
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 18, 70, 35)
                overtime = st.selectbox("Works Overtime?", df_viz['OverTime'].unique())
                marital_status = st.selectbox("Marital Status", df_viz['MaritalStatus'].unique())
            with col2:
                monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
                job_role = st.selectbox("Job Role", df_viz['JobRole'].unique())
                department = st.selectbox("Department", df_viz['Department'].unique())
            with col3:
                job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                total_working_years = st.slider("Total Working Years", 0, 40, 10)
                years_at_company = st.slider("Years at Company", 0, 40, 5)

            submitted = st.form_submit_button("Predict Attrition")

            if submitted:

                # 1. Create a dictionary from all user inputs.
                user_input = {
                    'Age': age, 'OverTime': overtime, 'MaritalStatus': marital_status,
                    'MonthlyIncome': monthly_income, 'JobRole': job_role, 'Department': department,
                    'JobSatisfaction': job_satisfaction, 'TotalWorkingYears': total_working_years,
                    'YearsAtCompany': years_at_company
                }

                # 2. Create a DataFrame from the dictionary.
                input_df = pd.DataFrame([user_input])

                # 3. One-hot encode the categorical features.
                input_df_encoded = pd.get_dummies(input_df)

                # 4. Align the columns with the training data.
                input_df_aligned = input_df_encoded.reindex(columns=model_features, fill_value=0)

                # 5. Scale the data using the loaded scaler.
                scaled_features = scaler.transform(input_df_aligned)

                # 6. Make the prediction.
                prediction_proba = attrition_model.predict_proba(scaled_features)
                confidence = np.max(prediction_proba) * 100

                st.subheader("Prediction Result")
                if attrition_model.predict(scaled_features)[0] == 1:
                    st.error(f"Prediction: Employee is **likely to leave** (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"Prediction: Employee is **likely to stay** (Confidence: {confidence:.2f}%)")
