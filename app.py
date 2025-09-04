import streamlit as st
import joblib
import os

# ==============================
# Load your trained model & vectorizer safely
# ==============================
vectorizer, model = None, None
error_message = None

try:
    if os.path.exists("vectorizer.pkl") and os.path.exists("model.pkl"):
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("model.pkl")
    else:
        error_message = "❌ Model files not found. Please ensure `vectorizer.pkl` and `model.pkl` are in the project folder."
except Exception as e:
    error_message = f"⚠️ Error loading model: {e}. This may be due to a version mismatch. Try retraining or checking requirements.txt."

# ==============================
# CSS Styling
# ==============================
st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #2e3b4e;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #6c757d;
            font-size: 20px;
            margin-bottom: 40px;
        }
        .result-card {
            padding: 20px;
            border-radius: 15px;
            color: white;
            font-size: 20px;
            text-align: center;
            margin-top: 20px;
        }
        .spam {
            background-color: #e63946;
        }
        .ham {
            background-color: #2a9d8f;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Title & subtitle
# ==============================
st.markdown('<div class="title">📧 Email Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect spam emails instantly with AI</div>', unsafe_allow_html=True)

# ==============================
# App Logic
# ==============================
if error_message:
    st.error(error_message)
else:
    email_text = st.text_area("Enter the email content below:", height=200)

    if st.button("🔍 Predict"):
        if email_text.strip() != "":
            try:
                # Transform and predict
                X = vectorizer.transform([email_text])
                prediction = model.predict(X)[0]

                # Show result
                if prediction == 1:  # Assuming 1 = spam
                    st.markdown('<div class="result-card spam">🚨 This email is Spam!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card ham">✅ This email is Safe (Ham)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Please enter some email text to analyze.")

# ==============================
# Footer
# ==============================
st.markdown("<br><hr><center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
