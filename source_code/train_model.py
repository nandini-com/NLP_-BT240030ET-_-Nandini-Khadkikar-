import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import preprocess_text


# Load dataset
data = pd.read_csv("../dataset/sentiment_dataset.csv")


# Text preprocessing
data["clean_text"] = data["text"].apply(preprocess_text)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)


# TF-IDF feature extraction
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# Predictions
y_pred = model.predict(X_test_tfidf)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model and vectorizer
joblib.dump(model, "../model/sentiment_model.joblib")
joblib.dump(vectorizer, "../model/tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved successfully.")
