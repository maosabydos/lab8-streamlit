from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

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

try:
    model = joblib.load(MODEL_DIR / "svc_model.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    questions = joblib.load(MODEL_DIR / "questions.pkl")
except FileNotFoundError as e:
    raise SystemExit(f"Model file not found: {e.filename}")


def predict_personality(*answers):
    if any(answer is None for answer in answers):
        raise gr.Error("Please answer all questions before submitting.")

    values = [LIKERT_OPTIONS[answer] for answer in answers]
    input_df = pd.DataFrame([values], columns=questions)

    prediction = model.predict(input_df)[0]
    persona = label_encoder.inverse_transform([prediction])[0]

    return f"## Predicted Personality: {persona}"


with gr.Blocks(title="Personality Prediction App") as demo:
    gr.Markdown(
        """
        # Personality Prediction App

        Rate the following statements on a scale from
        **Fully Disagree (-3)** to **Fully Agree (3)**.

        Please answer all questions before submitting.
        """
    )

    answer_components = [
        gr.Radio(
            choices=list(LIKERT_OPTIONS.keys()),
            label=f"{i + 1}. {question}",
            value=None,
        )
        for i, question in enumerate(questions)
    ]

    submit_btn = gr.Button("Predict Personality", variant="primary")
    result = gr.Markdown()

    submit_btn.click(
        fn=predict_personality,
        inputs=answer_components,
        outputs=result,
    )


if __name__ == "__main__":
    demo.launch()
