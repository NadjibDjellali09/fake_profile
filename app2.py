import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("LogisticRegression_model.pkl")

st.set_page_config(page_title="Fake Profile Detector", layout="centered")
st.title("🕵️‍♂️ Fake Profile Detection App")
st.markdown("Enter Instagram profile details to predict if it's **Fake** or **Real**.")

# Collect input features
flw = st.number_input("Number of Followers (flw)", min_value=0)
pos = st.number_input("Number of Posts (pos)", min_value=0)
lt = st.number_input("Total Likes (lt)", min_value=0)
erl = st.number_input("Engagement Rate - Likes (erl)", min_value=0)
erc = st.number_input("Engagement Rate - Comments (erc)", min_value=0)

# Calculate engineered features
followers_per_post = flw / (pos + 1)
likes_per_follower = lt / (flw + 1)
engagement_rate = (erl + erc) / (flw + 1)

# Construct input DataFrame for prediction
input_data = pd.DataFrame([{
    "flw": flw,
    "pos": pos,
    "lt": lt,
    "erl": erl,
    "erc": erc,
    "followers_per_post": followers_per_post,
    "likes_per_follower": likes_per_follower,
    "engagement_rate": engagement_rate
}])

# Predict and display result
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # Probability of class 1 (fake)
        
        if prediction == 1:
            st.error(f"⚠️ This is likely a **Fake Profile**. Confidence: {prob:.2%}")
        else:
            st.success(f"✅ This is likely a **Real Profile**. Confidence: {1 - prob:.2%}")
    except Exception as e:
        st.exception("❌ An error occurred during prediction. Make sure all fields are filled correctly.")
