# FakeJobClassification_App.py

import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# Streamlit UI
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("Enter a job description to check if it's real or fake.")

user_input = st.text_area("Job Description", height=200)

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    label = "🟢 Real Job" if prediction == 0 else "🔴 Fake Job"
    st.success(f"Prediction: {label}")
