import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load model, encoders, and selected features
# ----------------------------
@st.cache_resource
def load_assets():
    with open("churn_model_svm_top5.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    return model, encoders, selected_features

model, encoders, selected_features = load_assets()

# ----------------------------
# Page Config & Title
# ----------------------------
st.set_page_config(
    page_title="Telco Churn Predictor (Top 5 Features)",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Telco Customer Churn Prediction (Top 5 Features)")
st.markdown("""
    ✅ This model uses **only the top 5 most important features** selected by Recursive Feature Elimination (RFE)  
    🎯 Features: `SeniorCitizen`, `Dependents`, `OnlineSecurity`, `TechSupport`, `Contract`
""")

# ----------------------------
# Sidebar: Model Info & Dataset Stats
# ----------------------------
with st.sidebar:
    st.header("📊 Model & Dataset Info")
    
    # Dataset Churn Distribution
    st.subheader("Churn in Dataset")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie([5174, 1869], labels=["No Churn", "Churn"], autopct='%1.1f%%', startangle=90, colors=["#2E8B57", "#DC143C"])
    ax.axis('equal')
    st.pyplot(fig)
    st.caption("Based on Telco Customer Churn Dataset (7,043 customers)")

    # Model Performance
    st.subheader("Model Performance (Test Set)")
    st.metric("Accuracy", "70.8%")
    st.metric("Recall (Churn)", "74%")
    st.metric("Precision (Churn)", "47%")
    st.caption("Trained on SMOTE-balanced data")

    # Feature Importance (static)
    st.subheader("Top 5 Features")
    for i, feat in enumerate(selected_features, 1):
        st.write(f"{i}. `{feat}`")

# ----------------------------
# Main Input Section
# ----------------------------
st.header("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Build input DataFrame
input_data = pd.DataFrame({
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Dependents': [dependents],
    'OnlineSecurity': [online_security],
    'TechSupport': [tech_support],
    'Contract': [contract],
})

# Encode only categorical columns (SeniorCitizen is numeric)
for col in ['Dependents', 'OnlineSecurity', 'TechSupport', 'Contract']:
    input_data[col] = encoders[col].transform(input_data[col])

# ----------------------------
# Prediction & Results
# ----------------------------
if st.button("🔮 Predict Churn Risk", type="primary"):
    prediction = model.predict(input_data)[0]
    decision_score = model.decision_function(input_data)[0]

    # Display Input Summary
    st.subheader("📋 Input Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Senior Citizen", "Dependents", "Online Security", "Tech Support", "Contract"],
        "Value": [senior_citizen, dependents, online_security, tech_support, contract]
    })
    st.table(summary_df)

    # Prediction Result
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("🚨 **HIGH RISK OF CHURN**")
        st.info("Consider offering discounts, better support, or contract incentives.")
    else:
        st.success("✅ **LOW RISK — Customer likely to stay**")
        st.info("Great! Maintain service quality to retain this customer.")

    # Confidence Score
    st.write(f"**Model Confidence Score**: `{decision_score:.3f}`")
    st.caption("Higher positive → more confident in **CHURN**. Higher negative → more confident in **STAY**.")

    # Visualize Confidence
    fig, ax = plt.subplots(figsize=(6, 1))
    color = "red" if decision_score > 0 else "green"
    ax.barh(["Confidence"], [decision_score], color=color, height=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decision Function Score")
    ax.set_xlim(-2, 2)
    st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: SVM (RFE Top 5 Features) | Data: Telco Customer Churn Dataset")