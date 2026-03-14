# Task 1: End-to-End ML Project with Streamlit GUI


## **Problem Statement**

Customer churn occurs when a bank customer stops using the bank’s services. Losing customers directly affects revenue and increases acquisition costs.

The goal of this project is to:

* **Predict whether a customer will stay or leave** the bank based on demographic, financial, and account activity data.
  
* Identify the key factors driving churn.
* Deploy a real-time, interactive machine learning system that supports **data-driven retention strategies**.


## **Objective**

Build a complete machine learning system with an interactive **Streamlit GUI** to predict customer churn for a bank.

The project demonstrates the full pipeline from **data preprocessing, model training, evaluation, to deployment**, enabling real-time predictions and actionable business insights.


## **Dataset Overview**

The dataset contains information for **10,000 bank customers**.

**Features:**

* **Demographics:** Age, Gender, Geography
* **Financial:** CreditScore, Balance, EstimatedSalary, NumOfProducts
* **Account Activity:** Tenure, IsActiveMember, HasCrCard

**Target Variable:**

* `Exited` → 1 = churned, 0 = retained



## **Project Workflow**

### **1. Data Loading and Understanding**

* Load the dataset and inspect its structure and contents.
* Remove irrelevant or duplicate columns.

### **2. Data Preprocessing**

* Encode categorical variables using **Label Encoding / One-Hot Encoding**.
* Scale numerical features with **StandardScaler**.
* Split the dataset into training and test sets.

### **3. Exploratory Data Analysis (EDA)**

* Analyze distributions of numerical and categorical features.
* Identify patterns influencing churn using visualizations and plots.

### **4. Model Training and Evaluation**

* Train multiple models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting.
* Evaluate models using **accuracy, precision, recall, F1-score, and confusion matrix**.
* Compare performance in a results dataframe and select the best-performing model.

### **5. Model Saving**

* Save the selected model, scaler, and column transformer using **joblib**.

### **6. Streamlit GUI Development**

* Build an interactive interface for real-time predictions.
* GUI Features:

  * Input form for customer data
  * Prediction result with probability
  * Dataset preview
  * Model accuracy comparison
  * Feature importance chart


## **Streamlit UI**

![Input Form](https://github.com/user-attachments/assets/a3aca080-6a8a-42ab-8852-df59f578418a)

 ![Dataset Preview](https://github.com/user-attachments/assets/beb86870-c210-4385-a344-fe587e43b809)

![Model Performace Comparison](https://github.com/user-attachments/assets/97d43eea-4cd4-4df5-a0ef-a2cf357416d4)

![Feature Importance](https://github.com/user-attachments/assets/ad6d5b62-a0fa-4359-aecb-4e33473d025d)



## **Key Insights & Results**

### **Customer Profile**

* Older customers with higher balances are more likely to churn.
* Geography impacts churn: Germany shows the highest, France the lowest.
* Gender has minimal effect.

### **Products & Engagement**

* Customers with 3-4 products churn more than 1-2 product holders.
* Active members are less likely to leave.
* Credit card ownership has minimal impact.

### **Financial Factors**

* Credit scores and salaries are similar between churners and stayers.
* Balance shows a slight positive correlation with churn.

### **Model Performance**

The **Gradient Boosting model** was the best-performing model:

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 87%   |
| Precision (churn) | 0.79  |
| Recall (churn)    | 0.48  |
| F1-score (churn)  | 0.60  |

**Top Predictors:** Age, Balance, Number of Products

## **Business Recommendations**

* Target retention efforts on older, high-balance customers and those with multiple products.
  
* Strengthen engagement programs for inactive members.
* Focus marketing campaigns on regions with higher churn rates.
* Use predictive modeling to identify at-risk customers proactively.


## **Conclusion**

This project demonstrates a **complete end-to-end ML workflow**, integrating data analysis, multiple model training, evaluation, and deployment in a **Streamlit GUI**.

It enables **real-time churn prediction**, provides **insightful visualizations**, and supports **data-driven retention strategies** for banking operations.


