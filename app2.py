import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("LogisticRegression_model.pkl")

st.title("🔍 Fake Profile Detection")

st.write("Fill in the profile details to predict if it's likely a fake account.")

# Define form
with st.form("profile_form"):
    flg = st.checkbox("flg (e.g. has flag?)")
    cz = st.checkbox("cz")
    bl = st.checkbox("bl")
    pic = st.checkbox("pic (has profile picture?)")
    ni = st.checkbox("ni")
    lin = st.checkbox("lin")
    pr = st.checkbox("pr (private account?)")
    hc = st.checkbox("hc")
    fo = st.checkbox("fo")
    cs = st.checkbox("cs")
    cl = st.checkbox("cl")
    pi = st.checkbox("pi")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Create dataframe with all required fields
        input_data = pd.DataFrame([{
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

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"❌ Fake Profile Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Real Profile Detected! (Confidence: {1 - proba:.2%})")

    except Exception as e:
        st.error("❌ An error occurred during prediction. Please check your inputs.")
        st.exception(e)
