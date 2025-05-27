import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("LogisticRegression_model.pkl")

st.set_page_config(page_title="Fake Profile Detector", layout="centered")
st.title("🧠 Fake Profile Detector")

st.markdown("Fill in the profile metrics to check if it's a **fake** or **real** account.")

# Streamlit input form
with st.form("prediction_form"):
    # Numerical features
    flw = st.number_input("Followers (flw)", min_value=0)
    pos = st.number_input("Posts (pos)", min_value=0)
    lt = st.number_input("Lifetime in days (lt)", min_value=0)
    erc = st.number_input("Engagement Received Count (erc)", min_value=0)
    erl = st.number_input("Engagement Left Count (erl)", min_value=0)
    likes_per_follower = st.number_input("Likes per Follower", min_value=0.0, format="%.6f")
    engagement_rate = st.number_input("Engagement Rate", min_value=0.0, format="%.6f")
    followers_per_post = st.number_input("Followers per Post", min_value=0.0, format="%.6f")

    # Binary categorical features (checkbox = 0 or 1)
    flg = st.checkbox("Has Flag (flg)")
    cz = st.checkbox("Has Caption Zone (cz)")
    bl = st.checkbox("Has Bio Link (bl)")
    pic = st.checkbox("Has Profile Picture (pic)")
    ni = st.checkbox("Has Nickname (ni)")
    lin = st.checkbox("Has LinkedIn Link (lin)")
    pr = st.checkbox("Has Private Profile (pr)")
    hc = st.checkbox("Has Highlight Cover (hc)")
    fo = st.checkbox("Has Following List (fo)")
    cs = st.checkbox("Has Comment Section (cs)")
    cl = st.checkbox("Has Collaboration (cl)")
    pi = st.checkbox("Has Pinned Post (pi)")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([{
            'flw': flw,
            'pos': pos,
            'lt': lt,
            'erc': erc,
            'erl': erl,
            'likes_per_follower': likes_per_follower,
            'engagement_rate': engagement_rate,
            'followers_per_post': followers_per_post,
            'flg': int(flg),
            'cz': int(cz),
            'bl': int(bl),
            'pic': int(pic),
            'ni': int(ni),
            'lin': int(lin),
            'pr': int(pr),
            'hc': int(hc),
            'fo': int(fo),
            'cs': int(cs),
            'cl': int(cl),
            'pi': int(pi)
        }])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"❌ This is likely a **fake profile**. (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ This is likely a **real profile**. (Confidence: {(1 - proba):.2%})")

    except Exception as e:
        st.error("❌ An error occurred during prediction. Please check your inputs.")
        st.exception(e)
