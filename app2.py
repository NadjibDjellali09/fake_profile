import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("LogisticRegression_model.pkl")

# App title
st.title("🔍 Fake Profile Detection App")

# Description
st.write("Enter Instagram-like profile details to detect if it's likely **Fake** or **Real**.")

# Sidebar inputs
with st.form("profile_form"):
    flw = st.number_input("Followers (flw)", min_value=0)
    pos = st.number_input("Posts (pos)", min_value=0)
    lt = st.number_input("Likes total (lt)", min_value=0)
    erl = st.number_input("Engagement real likes (erl)", min_value=0)
    erc = st.number_input("Engagement real comments (erc)", min_value=0)

    submitted = st.form_submit_button("Predict")

# Process input and make prediction
if submitted:
    try:
        # Feature engineering (same as training)
        input_data = pd.DataFrame([{
            "flw": flw,
            "pos": pos,
            "lt": lt,
            "erl": erl,
            "erc": erc,
            "followers_per_post": flw / (pos + 1),
            "likes_per_follower": lt / (flw + 1),
            "engagement_rate": (erl + erc) / (flw + 1)
        }])[[
            "flw", "pos", "lt", "erl", "erc",
            "followers_per_post", "likes_per_follower", "engagement_rate"
        ]]

        # Predict
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        # Output result
        if prediction == 1:
            st.error(f"❌ Prediction: Likely a **FAKE** profile (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Prediction: Likely a **REAL** profile (Confidence: {1 - proba:.2%})")

    except Exception as e:
        st.error("❌ An error occurred during prediction. Make sure all fields are filled correctly.")
        st.exception(e)

