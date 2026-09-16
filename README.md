Tea Corner Website

## Student Details

Name:Nandini Khadkikar
Roll No / BT ID:BT240030ET
Branch:Electronics and Telecommunication Engineering   
Semester:V Semester
Course:Natural Language Processing (BT240030ET)

---

## 1. Problem Statement

People and organizations receive a large amount of textual feedback in the form of reviews, comments and messages. Manually identifying whether the expressed opinion is positive, negative or neutral can be time-consuming. This project develops an NLP-based application that automatically classifies text into Positive, Negative and Neutral sentiment categories.

## 2. Objective

The objective of this project is to develop an application-based Natural Language Processing system that preprocesses textual data, converts text into numerical features using TF-IDF and classifies sentiment using Logistic Regression.

## 3. Introduction

Sentiment Analysis is an important Natural Language Processing task used to identify the opinion or emotional polarity expressed in text. It can be used for analysing customer reviews, feedback, social media comments and other textual information.

This project implements a three-class sentiment classification system for Positive, Negative and Neutral text.

## 4. NLP Techniques Used

The project uses the following NLP and machine learning techniques:

* Text preprocessing
* Lowercase conversion
* Removal of unnecessary characters
* Whitespace normalization
* TF-IDF feature extraction
* Unigram and bigram features
* Logistic Regression classification

## 5. Dataset

The project uses a labelled sentiment dataset containing text samples belonging to three classes:

* Positive
* Negative
* Neutral

The dataset is stored inside the `dataset/` folder.

## 6. Software and Tools

* Python
* Google Colab
* Jupyter Notebook
* Pandas
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit
* Git
* GitHub

## 7. Methodology / Workflow

```text
Input Text
     ↓
Text Preprocessing
     ↓
Cleaned Text
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression
     ↓
Sentiment Prediction
     ↓
Positive / Negative / Neutral
```

## 8. Text Preprocessing

The input text is converted to lowercase and unnecessary characters and extra spaces are removed before feature extraction.

## 9. Model

Logistic Regression is used as the classification algorithm. TF-IDF is used to convert textual information into numerical feature vectors.

## 10. Steps to Execute

1. Open the Jupyter Notebook in Google Colab.
2. Install the required Python libraries.
3. Load the sentiment dataset.
4. Perform text preprocessing.
5. Split the dataset into training and testing data.
6. Apply TF-IDF vectorization.
7. Train the Logistic Regression model.
8. Evaluate the model using classification metrics.
9. Test the model with new input text.
10. Use the trained model for sentiment prediction.

## 11. Sample Input

```text
I really enjoyed this product.
```

## 12. Sample Output

```text
Predicted Sentiment: Positive
```

Another example:

```text
Input: This product is terrible and disappointing.
Predicted Sentiment: Negative
```

## 13. Results and Observations

The trained model successfully classifies input text into Positive, Negative and Neutral sentiment categories on the prepared dataset.

Performance is evaluated using accuracy, precision, recall, F1-score and a confusion matrix.

## 14. Conclusion

The project demonstrates an end-to-end Natural Language Processing workflow for sentiment classification. Text preprocessing and TF-IDF are used to prepare the textual data, while Logistic Regression performs the final classification.

The project also demonstrates how an NLP model can be organized, documented and maintained using GitHub.

## 15. Repository Structure

```text
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
├── source_code/
├── notebooks/
├── output/
├── screenshots/
└── model/
```

## 16. References

* Python Documentation
* Pandas Documentation
* Scikit-learn Documentation
* Streamlit Documentation
* Natural Language Processing learning resources
