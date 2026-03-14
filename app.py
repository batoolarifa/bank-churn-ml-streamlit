import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# load saved objects
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
ct = joblib.load("column_transformer.pkl")
results_df = pd.read_csv("result_df.csv")  # optional, for model accuracy display

# page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details below to predict churn probability:")

with st.form(key='churn_form'):
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 30)
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)
        balance = st.number_input("Balance", 0.0, 1_000_000.0, 10000.0, step=100.0)
        num_products = st.slider("Number of Products", 1, 4, 1)

    with col2:
        estimated_salary = st.number_input("Estimated Salary", 0.0, 1_000_000.0, 50000.0, step=1000.0)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])

    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    user_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_cr_card=="Yes" else 0],
        "IsActiveMember": [1 if is_active=="Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    # transform  and scale
    X_input = ct.transform(user_data)
    X_input = scaler.transform(X_input)

    # predict
    pred = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"The customer is likely to **churn** with probability {pred_prob*100:.1f}%")
    else:
        st.success(f"The customer is likely to **stay** with probability {(1-pred_prob)*100:.1f}%")

st.markdown("---")
st.subheader("Model Insights & Data Preview")

if st.checkbox("Show Dataset Preview"):
    df = pd.read_csv("dataset/Churn_Modelling.csv")
    st.dataframe(df.head(10))

# model performance comparison 
if st.checkbox("Show Model Accuracy Comparison"):
    st.dataframe(results_df)

# feature importance checkbox
if st.checkbox("Show Feature Importance"):
    try:
        feature_importance = model.feature_importances_
        feature_names = list(ct.get_feature_names_out())

        # clean the column names
        clean_feature_names = [name.split("__")[-1] for name in feature_names]

        fi_df = pd.DataFrame({
            "Feature": clean_feature_names,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(fi_df, 
                     x="Importance", 
                     y="Feature", 
                     orientation="h", 
                     text="Importance",
                     title="Feature Importance",
                     labels={"Importance": "Importance Score", "Feature": "Features"},
                     height=500)
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")