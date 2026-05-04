from util import preprocess_text, load_ai_comments, load_human_comments
from os import path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, f1_score, precision_score
import matplotlib.pyplot as plt

def split_data(X, y, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def apply_SMOTE(X_train, y_train):
    # X_vec = tfidf.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled

def apply_ADASYN(X_train, y_train):
    # X_vec = tfidf.fit_transform(X_train)
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    return X_resampled, y_resampled

def show_top_features(model, vectorizer, name, n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_pos = coefs.argsort()[-n:][::-1]
    top_neg = coefs.argsort()[:n]

    print(f"\n===== {name} =====")

    print(f"\nTop {n} AI indicators:")
    for i in top_pos:
        print(f"{feature_names[i]} ({coefs[i]:.4f})")

    print(f"\nTop {n} Human indicators:")
    for i in top_neg:
        print(f"{feature_names[i]} ({coefs[i]:.4f})")

def print_4_metrics(model, name, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def show_confusion_matrix(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Confusion Matrix:")
    print(cm)

def main():
    # Load data
    ai_comments_path = path.join(path.dirname(__file__), "youtube_ai_comments.csv")
    human_comments_path = path.join(path.dirname(__file__), "youtube_comments_1000_english.csv")

    ai_comments, ai_labels = load_ai_comments(ai_comments_path)
    human_comments, human_labels = load_human_comments(human_comments_path)

    # Combine data
    comments = ai_comments + human_comments
    labels = ai_labels + human_labels

    # Convert labels to binary
    y = np.array([1 if label == "ai" else 0 for label in labels])

    # Use raw text directly (NO preprocess_text here)
    X = comments

    # Vectorize text
    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_vec, y)

    # Train multiple models with different class balancing techniques

    # Model 1: Logistic Regression with balanced class weights
    model1 = LogisticRegression(max_iter=1000, class_weight="balanced")
    model1.fit(X_train, y_train)

    # Model 2: Logistic Regression with SMOTE
    X_train_smote, y_train_smote = apply_SMOTE(X_train, y_train)
    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X_train_smote, y_train_smote)

    # Model 3: Logistic Regression with ADASYN
    X_train_adasyn, y_train_adasyn = apply_ADASYN(X_train, y_train)
    model3 = LogisticRegression(max_iter=1000)
    model3.fit(X_train_adasyn, y_train_adasyn)

    # Evaluate each model
    show_top_features(model1, tfidf, "Balanced Weights")
    print_4_metrics(model1, "Balanced Weights", X_test, y_test)
    show_confusion_matrix(model1, "Balanced Weights", X_test, y_test)
    
    show_top_features(model2, tfidf, "SMOTE")
    print_4_metrics(model2, "SMOTE", X_test, y_test)
    show_confusion_matrix(model2, "SMOTE", X_test, y_test)

    show_top_features(model3, tfidf, "ADASYN")
    print_4_metrics(model3, "ADASYN", X_test, y_test)
    show_confusion_matrix(model3, "ADASYN", X_test, y_test)

    # Compare ROC curves
    y_probs1 = model1.predict_proba(X_test)[:, 1]
    y_probs2 = model2.predict_proba(X_test)[:, 1]
    y_probs3 = model3.predict_proba(X_test)[:, 1]

    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_probs1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_probs2)
    fpr3, tpr3, thresholds3 = roc_curve(y_test, y_probs3)

    plt.plot(fpr1, tpr1, label="Balanced Weights")
    plt.plot(fpr2, tpr2, label="SMOTE")
    plt.plot(fpr3, tpr3, label="ADASYN")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()



if __name__ == "__main__":
    main()
