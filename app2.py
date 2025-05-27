import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("LogisticRegression_model.pkl")

st.set_page_config(page_title="Fake Profile Detector", layout="centered")
st.title("🤖 Fake Profile Detection App")
st.write("Enter the profile metrics below to check if it's a **fake profile**.")

# Form for input
with st.form("prediction_form"):
    flw = st.number_input("Number of Followers (flw)", min_value=0)
    pos = st.number_input("Number of Posts (pos)", min_value=0)
    lt = st.number_input("Lifetime (in days) (lt)", min_value=0)
    erc = st.number_input("Engagement received count (erc)", min_value=0)
    erl = st.number_input("Engagement left count (erl)", min_value=0)
    likes_per_follower = st.number_input("Likes per Follower", min_value=0.0, format="%.6f")
    engagement_rate = st.number_input("Engagement Rate", min_value=0.0, format="%.6f")
    followers_per_post = st.number_input("Followers per Post", min_value=0.0, format="%.6f")

    submitted = st.form_submit_button("Predict")

# When form is submitted
if submitted:
    try:
        input_df = pd.DataFrame([{
            "flw": flw,
            "pos": pos,
            "lt": lt,
            "erc": erc,
            "erl": erl,
            "likes_per_follower": likes_per_follower,
            "engagement_rate": engagement_rate,
            "followers_per_post": followers_per_post
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"❌ This is likely a **fake profile**. (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ This appears to be a **real profile**. (Confidence: {(1 - probability):.2%})")

    except Exception as e:
        st.error("❌ An error occurred during prediction. Please check your inputs.")
        st.exception(e)

