from util import preprocess_text
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def split_data(X, y, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def SMOTE(X_train, y_train):
    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_resample(X_train)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_vec, y_train)
    return X_resampled, y_resampled


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data.csv")
    X = df["text"].values
    y = df["label"].values

    # Preprocess text data
    X = np.array([preprocess_text(text) for text in X])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = SMOTE(X_train, y_train)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model on the test set
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
