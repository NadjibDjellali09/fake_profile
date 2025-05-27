import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("RandomForest_model.pkl")

# Title
st.title("🕵️‍♂️ Fake Profile Detector")
st.markdown("Enter profile statistics to predict if it's a fake account.")

# Input form
with st.form("profile_form"):
    flw = st.number_input("Number of Followers", min_value=0)
    flg = st.number_input("Number of Following", min_value=0)
    pos = st.number_input("Number of Posts", min_value=0)
    lt = st.number_input("Number of Likes Total", min_value=0)
    cmt = st.number_input("Number of Comments Total", min_value=0)
    erl = st.number_input("Likes per post (avg)", min_value=0.0)
    erc = st.number_input("Comments per post (avg)", min_value=0.0)
    
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Feature engineering (same as training)
    followers_per_post = flw / (pos + 1)
    likes_per_follower = lt / (flw + 1)
    engagement_rate = (erl + erc) / (flw + 1)

    # Create DataFrame
    input_data = pd.DataFrame([{
        "flw": flw,
        "flg": flg,
        "pos": pos,
        "lt": lt,
        "cmt": cmt,
        "erl": erl,
        "erc": erc,
        "followers_per_post": followers_per_post,
        "likes_per_follower": likes_per_follower,
        "engagement_rate": engagement_rate
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.error(f" Fake Profile Detected with probability {prob:.2f}")
    else:
        st.success(f" Real Profile Detected with probability {1 - prob:.2f}")
