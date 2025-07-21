import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set Streamlit page config
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# App title
st.title("💼 Employee Salary Prediction")
st.markdown("Predict salary based on personal and job-related factors using a trained ML model.")

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select a Model", [
    "Random Forest", "XGBoost", "Linear Regression"
])

# Load model paths
model_paths = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Linear Regression": "linear_model.pkl"
}

# Load selected model
model_file = model_paths[selected_model_name]

if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found.")
    st.stop()

model = joblib.load(model_file)

# Load encoders
if not os.path.exists("encoders.pkl"):
    st.error("Encoders file not found.")
    st.stop()

encoders = joblib.load("encoders.pkl")

# Input form
with st.form("input_form"):
    st.subheader("📥 Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, step=1, value=30)
        gender = st.selectbox("Gender", options=encoders['Gender'].classes_)

    with col2:
        education = st.selectbox("Education Level", options=encoders['Education Level'].classes_)
        job_title = st.selectbox("Job Title", options=encoders['Job Title'].classes_)
        experience = st.slider("Years of Experience", min_value=0, max_value=40, step=1, value=5)

    submitted = st.form_submit_button("🚀 Predict Salary")

# Validate and predict
if submitted:
    if age < 18 or experience < 0:
        st.error("Please enter valid inputs. Age must be ≥ 18 and experience ≥ 0.")
    else:
        # Encode inputs
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'Education Level': encoders['Education Level'].transform([education])[0],
            'Job Title': encoders['Job Title'].transform([job_title])[0],
            'Years of Experience': experience
        }])

        # Predict
        predicted_salary = model.predict(input_data)[0]
        st.success(f"💰 Estimated Salary: **₹ {predicted_salary:,.2f}**")
        st.info(f"🔍 Prediction made using **{selected_model_name}** model.")

        with st.expander("📄 Show Encoded Input Data"):
            st.write(input_data)

st.markdown("---")

# Model Comparison Table
with st.expander("📊 View Model Evaluation Matrix"):
    try:
        df = pd.read_csv("model_evaluation_summary.csv")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Matrix as CSV",
            data=df.to_csv(index=False),
            file_name="model_evaluation_summary.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning("Model evaluation summary file not found.")

# Footer
st.markdown("---")
st.markdown("✅ Built with Streamlit | 📈 Compare ML models | 🔍 Live predictions")
