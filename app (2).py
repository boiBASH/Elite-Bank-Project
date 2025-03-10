import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("best_model.pkl")

# Custom CSS for Dark Mode Theme
st.markdown(
    """
    <style>
        body {background-color: #121212; color: white;}
        .stApp {background-color: #1e1e1e;}
        .stTitle, .stText, .stHeader, .stSubheader {color: white;}
        .stButton > button {background-color: #ff4b4b; color: white; font-size: 16px; padding: 10px 24px; border-radius: 8px;}
        .stNumberInput > div > input {background-color: #2e2e2e; color: white; border-radius: 5px;}
        .stSelectbox, .stRadio {color: white;}
        .stSlider > div > div > div > div {background-color: #ff4b4b;}
        .result-positive {color: #4CAF50; font-weight: bold; font-size: 18px;}
        .result-negative {color: #FF5733; font-weight: bold; font-size: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ FIXED: Streamlit Sidebar with Updated Image Parameter
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Logo_BANK.png", use_container_width=True)
st.sidebar.title("🔍 About This App")
st.sidebar.info("This app predicts if a customer is likely to become a long-term investor using a machine learning model.")

# App Header
st.title("🌙📈 Long-Term Investor Prediction App")
st.write("Fill in the details below to predict whether a customer is likely to invest.")

# Create input columns for better UI alignment
col1, col2 = st.columns(2)

# User Inputs (Left Column)
with col1:
    age = st.slider("🧑 Age", 18, 100, 35)
    balance = st.number_input("💰 Balance (€)", -10000, 100000, 1000, step=500)
    duration = st.slider("📞 Call Duration (seconds)", 0, 5000, 300, step=10)
    campaign = st.slider("📢 Number of Contacts in Campaign", 1, 50, 3)
    pdays = st.number_input("📅 Days Since Last Contact", -1, 1000, 30)
    previous = st.slider("🔁 Previous Campaign Contacts", 0, 50, 1)
    day = st.slider("📆 Last Contact Day of Month", 1, 31, 15)

# Categorical Inputs (Right Column)
with col2:
    job = st.selectbox("👔 Job", ["admin.", "technician", "services", "management", "retired", "blue-collar", "unemployed", "entrepreneur", "housemaid", "student", "self-employed"])
    marital = st.radio("💍 Marital Status", ["married", "single", "divorced"])
    education = st.radio("🎓 Education", ["primary", "secondary", "tertiary"])
    default = st.radio("💳 Credit Default", ["yes", "no"])
    housing = st.radio("🏡 Housing Loan", ["yes", "no"])
    loan = st.radio("🏦 Personal Loan", ["yes", "no"])
    contact = st.radio("☎️ Contact Type", ["cellular", "telephone"])
    month = st.selectbox("📆 Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = st.radio("📊 Previous Campaign Outcome", ["success", "failure", "other", "unknown"])

# Convert input data into a DataFrame
input_data = pd.DataFrame([[age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]],
                          columns=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])

# Add Predict Button with Styling
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("🔮 Predict Investment Probability"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Display Results with Colors and Icons
    if prediction == 1:
        st.success(f"✅ The customer is likely to invest! Probability: **{probability:.2f}**")
        st.markdown('<p class="result-positive">🎉 Great! This customer has a high chance of investing.</p>', unsafe_allow_html=True)
    else:
        st.error(f"❌ The customer is unlikely to invest. Probability: **{probability:.2f}**")
        st.markdown('<p class="result-negative">⚠️ This customer may need more persuasion or a better offer.</p>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Streamlit & Machine Learning")

