import pandas as pd
from util import preprocess_text, load_ai_comments, load_human_comments


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
   
    #1 Load AI-generated comments and assign "ai" labels
    ai_comments, ai_labels = load_ai_comments("youtube_ai_comments.csv")
    
    # Load human comments and assign "human" labels
    human_comments, human_labels = load_human_comments("youtube_comments_1000_english.csv")

   
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

  
    #5 Convert text into numerical features using TF-IDF
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



if __name__ == "__main__":
    main()
