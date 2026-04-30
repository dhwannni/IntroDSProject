import pandas as pd
from util import preprocess_text, load_ai_comments, load_human_comments

def main():
    # 1. Load both datasets
    ai_comments, ai_labels = load_ai_comments("youtube_ai_comments.csv")
    human_comments, human_labels = load_human_comments("youtube_comments_1000_english.csv")

    # 2. Combine comments and labels
    comments = ai_comments + human_comments
    labels = ai_labels + human_labels

    # 3. Put into one dataframe
    df = pd.DataFrame({
        "label": labels,
        "comment": comments
    })

    # 4. Preprocess comments
    df["tokens"] = df["comment"].apply(
        lambda text: preprocess_text(
            text,
            remove_stopwords=False,
            remove_emojis=False,
            remove_punctuation=False
        )
    )

    # 5. Save combined dataset
    df.to_csv("combined_ai_human_comments.csv", index=False)

    print(df.head())
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
