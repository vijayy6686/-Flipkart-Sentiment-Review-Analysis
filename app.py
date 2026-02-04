# =========================================
# STREAMLIT SENTIMENT ANALYSIS APP
# =========================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================================
# NLTK DOWNLOAD (safe)
# =========================================
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# =========================================
# LOAD MODEL & VECTORIZER
# =========================================
model = joblib.load("model.pkl")      # classifier
vectorizer = joblib.load("tfidf.pkl")  # tfidf

# =========================================
# TEXT PREPROCESSING (must match training)
# =========================================
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review and predict its sentiment")

review_text = st.text_area(
    "Enter your review here:",
    height=150,
    placeholder="Example: This product quality is amazing and worth the price!"
)

# =========================================
# PREDICTION
# =========================================
if st.button("Predict Sentiment"):

    if not review_text.strip():
        st.warning("Please enter a review text.")

    else:
        # step 1: preprocess
        clean_text = preprocess(review_text)

        # step 2: vectorize (IMPORTANT → 2D sparse matrix)
        X = vectorizer.transform([clean_text])

        # step 3: predict
        prediction = model.predict(X)[0]

        # safer mapping using classes
        classes = model.classes_

        if prediction == classes[1]:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("**Model:** TF-IDF + Logistic Regression")
st.markdown("**Metric Used:** F1-score")
