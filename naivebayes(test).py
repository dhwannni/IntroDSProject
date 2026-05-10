from os import path

import pandas as pd
from util import preprocess_text, load_ai_comments, load_human_comments


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


#GRAPH FUNCTIONS 


def show_prediction_counts(y_pred):
    # Count how many predictions were "ai" vs "human"
    counts = pd.Series(y_pred).value_counts()

    counts.plot(kind="bar")
    plt.xlabel("Predicted Label")
    plt.ylabel("Number of Comments")
    plt.title("Naive Bayes Prediction Counts")
    plt.xticks(rotation=0)
    plt.show()


def show_actual_counts(y_test):
    # Count how many actual test labels were "ai" vs "human"
    counts = y_test.value_counts()

    counts.plot(kind="bar")
    plt.xlabel("Actual Label")
    plt.ylabel("Number of Comments")
    plt.title("Actual Test Label Counts")
    plt.xticks(rotation=0)
    plt.show()


def show_confusion_matrix_graph(y_test, y_pred):
    # Labels are ordered so rows/columns are easier to read
    labels = ["human", "ai"]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.imshow(cm)
    plt.title("Naive Bayes Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    # Put the numbers inside each box
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.show()


def show_roc_curve(model, X_test_vec, y_test):
    # Convert labels to binary because roc_curve needs 0/1 labels
    # ai = 1, human = 0
    y_test_binary = y_test.map(lambda label: 1 if label == "ai" else 0)

    # Get probability that each comment is AI
    ai_index = list(model.classes_).index("ai")
    y_probs = model.predict_proba(X_test_vec)[:, ai_index]

    fpr, tpr, thresholds = roc_curve(y_test_binary, y_probs)

    plt.plot(fpr, tpr, label="Naive Bayes")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Naive Bayes ROC Curve")
    plt.legend()
    plt.show()




def main():
   
    #1 Load data
    ai_comments_path = path.join(path.dirname(__file__), "youtube_ai_comments.csv")
    human_comments_path = path.join(path.dirname(__file__), "youtube_comments_1000_english.csv")

    ai_comments, ai_labels = load_ai_comments(ai_comments_path)
    human_comments, human_labels = load_human_comments(human_comments_path)

   
    #2  Merge both datasets into one list
    comments = ai_comments + human_comments
    labels = ai_labels + human_labels

    df = pd.DataFrame({
        "label": labels,
        "comment": comments
    })

    #3 Convert each comment into tokens, then join back into a cleaned string
    # (Vectorizers expect text, not token lists)
    df["processed"] = df["comment"].apply(
        lambda text: " ".join(preprocess_text(
            text,
            remove_stopwords=False,   
            remove_emojis=False,      
            remove_punctuation=False  
        ))
    )

    # Features (X) = processed text
    X = df["processed"]
    
    # Labels (y) = "ai" or "human"
    y = df["label"]

    
    #4 Split data into training (75%) and testing (25%)
    # stratify=y makes equal proportion of ai/human in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=2026,
        stratify=y
    )

  
    #5 Convert text into numerical features using CountVectorizer
    vectorizer = CountVectorizer()

    # Fit on training data and transform both train/test
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

   
    #6 MultinomialNB works well for text classification
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

  
    # 7. predict
    y_pred = model.predict(X_test_vec)

    print("Prediction counts:")
    print(pd.Series(y_pred).value_counts())

    print("\nActual counts:")
    print(y_test.value_counts())
    
    #8 Print standard classification metrics
    print("Naive Bayes Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label="ai", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, pos_label="ai", zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label="ai", zero_division=0))

    # Confusion matrix -  detailed breakdown of predictions
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

     #9 Graphs
    show_prediction_counts(y_pred)
    
    show_actual_counts(y_test)
    show_confusion_matrix_graph(y_test, y_pred)
    show_roc_curve(model, X_test_vec, y_test)



if __name__ == "__main__":
    main()
