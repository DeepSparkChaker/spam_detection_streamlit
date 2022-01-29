# -*- coding: utf-8 -*-
"""Preparation class"""
import numpy as np
import pandas as pd
import os 
import re # regex library
# Read the Data
# Train, Test Split
from sklearn.model_selection import train_test_split
# Training a Neural Network Pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump

# Preprocess Heleper 
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
# Predict Function 
def classify_message(model, message):

	message = preprocessor(message)
	label = model.predict([message])[0]
	spam_prob = model.predict_proba([message])

	return {'label': label, 'spam_probability': spam_prob[0][1]}
