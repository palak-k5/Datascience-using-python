# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:11:05 2023

@author: hp
"""

# 6_1 Naive Bayes classification
# Demonstrate application of Naive Bayes using python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

corpus = [
    ("I love Python", "positive"),
    ("Naive Bayes is interesting", "positive"),
    ("I dislike bugs", "negative"),
    ("Machine learning is fascinating", "positive"),
    ("I hate errors", "negative")
]

X, y = zip(*corpus)
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(report)