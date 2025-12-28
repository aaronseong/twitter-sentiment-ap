from pathlib import Path
import streamlit as st
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

tfidf = joblib.load(BASE_DIR / "tfidf.pkl")
model = joblib.load(BASE_DIR / "best_model.pkl")

st.title("Twitter Sentiment Classification (TF-IDF + Logistic Regression)")
st.caption("Multiclass: Negative / Neutral / Positive")

user_text = st.text_area("Enter your sentence:")

# Optional threshold: if confidence too low, treat as neutral/uncertain
CONF_THRESHOLD = 0.55

if st.button("Predict"):
    text = user_text.strip()
    if not text:
        st.warning("Please enter some text!")
    else:
        X = tfidf.transform([text.lower()])
        pred = model.predict(X)[0]

        # Show confidence if model supports probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            best_idx = int(np.argmax(probs))
            best_class = classes[best_idx]
            best_prob = float(probs[best_idx])

            final_pred = best_class
            if best_prob < CONF_THRESHOLD:
                final_pred = "neutral"  # or "uncertain"

            st.success(f"Prediction: **{final_pred}** (confidence: {best_prob:.2f})")

            st.write("Class probabilities:")
            for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
                st.write(f"- {c}: {p:.2f}")
        else:
            st.success(f"Prediction: **{pred}**")

