# !streamlit run app.py --server.headless true

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent.parent / "modeling"

LIKERT_OPTIONS = {
    "Fully Agree": 3,
    "Partially Agree": 2,
    "Slightly Agree": 1,
    "Neutral": 0,
    "Slightly Disagree": -1,
    "Partially Disagree": -2,
    "Fully Disagree": -3,
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "svc_model.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    questions = joblib.load(MODEL_DIR / "questions.pkl")
    return model, label_encoder, questions


st.set_page_config(page_title="Personality Prediction", layout="wide")

try:
    model, label_encoder, questions = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e.filename}")
    st.stop()

st.title("Personality Prediction App")

st.markdown(
    "Rate the following statements on a scale from "
    "**Fully Disagree (-3)** to **Fully Agree (3)**."
)
st.divider()

st.subheader("Personality Test")
st.write("Please answer all questions before submitting.")

with st.form("personality_form"):
    responses = []
    for i, question in enumerate(questions):
        responses.append(
            st.radio(
                f"**{i + 1}. {question}**",
                options=list(LIKERT_OPTIONS.keys()),
                index=None,
                key=f"question_{i}",
            )
        )

submitted = st.form_submit_button("Predict Personality", use_container_width=True)

if submitted:
    if any(r is None for r in responses):
        st.error("Please answer all the questions before submitting.")
    else:
        values = [LIKERT_OPTIONS[r] for r in responses]
        input_df = pd.DataFrame([values], columns=questions)

        prediction = model.predict(input_df)[0]
        persona = label_encoder.inverse_transform([prediction])[0]

        st.success("Personality Prediction")
        st.markdown(f"# {persona}")
        st.balloons()