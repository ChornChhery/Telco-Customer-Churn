import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and encoders
@st.cache_resource
def load_model():
    with open("churn_model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# Page config
st.set_page_config(page_title="Telco Churn Predictor (XGBoost)", page_icon="📈")
st.title("📞 Telco Customer Churn Prediction")
st.markdown("""
    **Model: XGBoost (Trained on 80% data with SMOTE)**  
    High accuracy and good recall. Fill in customer details below.
""")

st.header("📝 Customer Information")

# Input form in 2 columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.01)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=0.01)

# Predict button
if st.button("🔮 Predict Churn"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Encode categorical columns — ONLY if column exists
    for col in encoders.keys():
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            st.warning(f"⚠️ Column '{col}' not found in input. Skipping encoding.")

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display result
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 Customer is likely to **CHURN** (Probability: {probability[1]:.2%})")
    else:
        st.success(f"✅ Customer is likely to **STAY** (Probability: {probability[0]:.2%})")

    # Show probability chart
    st.subheader("📊 Probability Distribution")
    prob_data = pd.DataFrame({
        'Class': ['Stay (No)', 'Churn (Yes)'],
        'Probability': [probability[0], probability[1]]
    }).set_index('Class')
    st.bar_chart(prob_data)

# Footer
st.markdown("---")
st.subheader("🧠 Model Info")
st.markdown("""
- ✅ **Algorithm**: XGBoost
- ✅ **Train/Test Split**: 80% / 20%
- ✅ **Class Balancing**: SMOTE applied on training set
- ✅ **Performance**: ~79% Accuracy, ~60% F1-Score for Churn
""")

st.caption("Built with ❤️ using Streamlit | Model: XGBoost | Data: Telco Customer Churn")