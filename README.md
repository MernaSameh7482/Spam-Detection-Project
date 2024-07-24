# Spam Classification Project

## Overview
This project aims to classify text messages as spam or not spam using various machine learning models including LSTM, RNN, KNN, and Naive Bayes. The dataset used is the SMS Spam Collection Dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Dependencies](#dependencies)

## Dataset
The dataset used in this project is the SMS Spam Collection Dataset, which is a set of SMS messages labeled as either spam or ham (not spam).

## Preprocessing
The preprocessing steps include:
1. Removing special characters, digits, and punctuation.
2. Tokenization and lemmatization.
3. Removing stopwords.
4. Handling imbalanced classes by upsampling the minority class.

## Models
The following models were used for classification:
1. **LSTM (Long Short-Term Memory)**
   - Architecture: Embedding layer, LSTM layer, Dense layer with sigmoid activation.
   - Training: Adam optimizer, binary cross-entropy loss.

2. **RNN (Recurrent Neural Network)**
   - Architecture: Embedding layer, SimpleRNN layer, Dense layer with sigmoid activation.
   - Training: Adam optimizer, binary cross-entropy loss.

3. **KNN (K-Nearest Neighbors)**
   - Hyperparameter tuning using GridSearchCV.
   - Parameters: Number of neighbors, weighting scheme, power parameter for Minkowski distance.

4. **Naive Bayes**
   - Model: Multinomial Naive Bayes.

## Results
- **LSTM Model Accuracy:**
  - Train: 99.8%
  - Test: 99.5%
  
- **RNN Model Accuracy:**
  - Train: 99.8%
  - Test: 99.4%
  
- **KNN Model Accuracy:**
  - Train: 99%
  - Test: 95%
  
- **Naive Bayes Model Accuracy:**
  - Train: 76%
  - Test: 78%

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Seaborn
- Matplotlib
- WordCloud
- Emoji

Install the dependencies using:
```bash
pip install numpy pandas scikit-learn tensorflow keras nltk seaborn matplotlib wordcloud emoji
