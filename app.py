import streamlit as st
import joblib

# Load TF-IDF and model
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("best_model.pkl")

st.title("Twitter Sentiment Prediction")

user_text = st.text_area("Enter your sentence:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform text → vector
        X = tfidf.transform([user_text])
        
        # Predict
        pred = model.predict(X)[0]

        st.success(f"Prediction: **{pred}**")
