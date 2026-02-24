import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Stopwords (keep negations)
stop_words = set(stopwords.words('english'))
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("🍽 Restaurant Review Sentiment Analyzer")
st.write("Enter a review to predict sentiment")

user_input = st.text_area("Type your review here...")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😡")
    else:
        st.warning("Please enter a review.")