import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from utils import preprocess_text

DATA_PATH = "../dataset/sentiment_dataset.csv"
MODEL_PATH = "../model/sentiment_model.joblib"

df = pd.read_csv(DATA_PATH)

df["clean_text"] = df["text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

print(classification_report(y_test, predictions))

os.makedirs("../model", exist_ok=True)

joblib.dump(
    (vectorizer, model),
    MODEL_PATH
)

print("Model saved successfully.")
