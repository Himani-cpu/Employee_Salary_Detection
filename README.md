# 💼 Employee Salary Prediction App


This is a **Streamlit web application** to predict the **Salary of an Employee** based on job and personal attributes like age, gender, job title, education level, and experience. It uses **Multiple Regression Models** including:

✅ Random Forest Regressor

✅ XGBoost Regressor

✅ Linear Regression


## 🚀 Features

🔢 Input employee details via UI

📈 Switch between different ML models

💰 Predict salary instantly

📊 Compare model performance (R² Score, MAE, RMSE)

📄 Download evaluation summary CSV

✅ Encoded input data preview

🧠 Trained model loading with `.pkl` files

🧠 Uses pre-saved label encoders



## 📂 Project Structure

 Employee_Salary_Detection/
 
 │
 
 ├── main.py # Main Streamlit app
 
 ├── random_forest_model.pkl # Trained Random Forest model
 
 ├── xgboost_model.pkl # Trained XGBoost model
 
 ├── linear_model.pkl # Trained Linear Regression model
 
 ├── encoders.pkl # Contains all label encoders
 
 ├── model_evaluation_summary.csv # Evaluation matrix with R², MAE, RMSE
 
 ├── requirements.txt # All required Python packages


## ▶️ How to Run the App

### 1. Clone the repository or move to the app folder

cd Employee_Salary_Detection

### 2. Install Required Packages

pip install -r requirements.txt

### 3. Run Streamlit

streamlit run salary_prediction_app.py

## 💻 App Inputs
Age: Integer [18–65]

Gender: Male / Female (Label encoded)

Education Level: Bachelor / Master / PhD (Label encoded)

Job Title: Select from trained list (Label encoded)

Years of Experience: Integer [0–40]

## 📊 Model Comparison Output

You’ll find a downloadable table for all models tested, showing:

R² Score (Goodness of Fit)

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

## 📁 Notes

Ensure .pkl files (models and encoders) are in the same directory as the app.

You must include all class labels in encoders.pkl used during training.




