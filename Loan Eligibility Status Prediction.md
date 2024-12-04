** Loan Eligibility Status Prediction

Project Overview:
The Loan Eligibility Prediction System is a supervised learning binary classification project designed to automate the loan eligibility process for customers applying to Dream Housing Finance. The system leverages machine learning algorithms to predict loan eligibility based on customer details, enabling the company to efficiently target eligible customer segments.

🔍 Problem Statement:
Dream Housing Finance wants to automate the loan eligibility process based on customer information such as:
Gender
Marital Status
Education
Number of Dependents
Income
Loan Amount
Credit History
Property Area

The goal is to identify customers who are eligible for a loan in real-time, based on their provided details during application.

📁 Dataset:
The dataset contains information about past applicants, including their demographic, financial, and credit details. Key columns include:
Gender
Married
Education
Dependents
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
Loan_Status (Target)

🔧 Project Workflow:
Data Understanding and Cleaning
Handled missing values.
Encoded categorical variables.
Scaled numerical features.
Exploratory Data Analysis (EDA)
Visualized relationships between features and loan eligibility.
Identified key factors influencing loan approval.

Model Building:
Applied multiple machine learning algorithms:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest
AdaBoost
XGBoost

Performed hyperparameter tuning for optimal performance.
XGBoost provided the highest accuracy.
Model Saving

🚀 Deployment-Ready Features
The trained XGBoost model is saved and can be loaded for predictions using:
python
Copy code
import joblib
model = joblib.load("loan_eligibility_model.joblib")
prediction = model.predict(input_data)

📈 Key Insights:
Credit History and Applicant Income are significant predictors of loan eligibility.
XGBoost outperformed other models due to its ability to handle non-linear relationships and imbalanced datasets.

🛠️ Tools and Technologies Used:
Programming Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib
Model Saving: Joblib
IDE: Jupyter Notebook

🌟 Future Enhancements:
Deploy the model as a web application using Flask or Django.
Integrate the system with a user-friendly front-end for real-time predictions.
Improve model performance by collecting more data or exploring advanced algorithms.

Project By - Rahul Ugale