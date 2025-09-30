import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
@st.cache_resource
def load_model():
    with open("churn_model_svm_top5.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# Page config
st.set_page_config(page_title="Telco Churn Predictor (Top 5 Features)", page_icon="📞")
st.title("📞 Telco Customer Churn Prediction (Top 5 Features)")
st.markdown("""
    ✅ Model uses **only the top 5 most important features** selected by RFE  
    🎯 Features: `SeniorCitizen`, `Dependents`, `OnlineSecurity`, `TechSupport`, `Contract`
""")

st.header("📝 Enter Customer Details")

# --- INPUT FORM with only TOP 5 features ---
col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Churn Risk"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Dependents': [dependents],
        'OnlineSecurity': [online_security],
        'TechSupport': [tech_support],
        'Contract': [contract],
    })

    # Encode categorical features
    for col in encoders.keys():
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("🚨 **HIGH RISK OF CHURN**")
    else:
        st.success("✅ **LOW RISK — Customer likely to stay**")

    # Confidence score
    decision_score = model.decision_function(input_data)[0]
    st.write(f"**Model Confidence Score**: {decision_score:.3f}")
    st.caption("Higher positive = more confident in CHURN. Higher negative = more confident in STAY.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: SVM | Features: Top 5 Selected by RFE")
