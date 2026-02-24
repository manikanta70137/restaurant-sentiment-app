# 🍽️ Restaurant Review Sentiment Analyzer

A Machine Learning web application that predicts whether a restaurant review is **Positive** or **Negative** using Natural Language Processing (NLP).

🚀 **Live Demo:**  
https://restaurant-sentiment-app-adhcwmbjuzxaiqeoz3rtkf.streamlit.app/

---

## 📌 Project Overview

This project uses:

- TF-IDF Vectorization
- Naive Bayes Classifier
- Streamlit for Web Deployment
- NLTK for Text Preprocessing

The application allows users to enter a restaurant review and instantly get a sentiment prediction.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy

---

## ⚙️ How It Works

1. User enters a restaurant review.
2. Text is preprocessed (cleaning, tokenization, stopword removal).
3. TF-IDF converts text into numerical features.
4. Trained Naive Bayes model predicts sentiment.
5. Result is displayed instantly on the web app.

---

## 📊 Model Performance

- Accuracy: ~81%+
- Precision: ~84%
- Recall: ~76%
- F1 Score: ~78%

*(Metrics may vary slightly based on dataset split.)*

---
<img width="1745" height="1048" alt="image" src="https://github.com/user-attachments/assets/7e0e2c4a-4eee-4722-a314-63a0d7286d6e" />
<img width="1712" height="1022" alt="image" src="https://github.com/user-attachments/assets/222802dc-f853-4ec9-b15d-2e8c1ea0274f" />



## 📁 Project Structure
```
restaurant-sentiment-app/
│
├── app.py                         # Streamlit web application
├── Dataset.csv                    # Training dataset
├── sentiment_model.pkl            # Trained Naive Bayes model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── requirements.txt               # Project dependencies
├── restaurant-sentiment-app.ipynb # Model training notebook
│
└── .ipynb_checkpoints/            # Jupyter auto-generated files (can be ignored)
```
---

## 🚀 Deployment

This app is deployed on **Streamlit Cloud**.

To deploy your own version:

1. Fork this repository
2. Connect GitHub to Streamlit Cloud
3. Select `app.py` as main file
4. Deploy 🚀

---

## 🧠 Future Improvements

- Add Neutral sentiment class
- Improve model accuracy
- Add prediction confidence score
- Store user review history
- Improve UI design

---

## 👨‍💻 Author

**Manikanta Chowdary**  
Machine Learning & Data Enthusiast

---

## ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!
