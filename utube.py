import csv
import string

input_file = "youtube_comments_cleaned.csv"
output_file = "youtube_comments_1000_english.csv"

seen = set()
count = 0

def looks_english(text):
    letters = sum(c in string.ascii_letters for c in text)
    return letters / max(len(text), 1) > 0.6   # 60% letters rule

with open(input_file, newline='', encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    with open(output_file, "w", newline='', encoding="utf-8") as outfile:
        fieldnames = ["VideoID", "VideoTitle", "CommentText", "PublishedAt"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:

            date = row["PublishedAt"].strip()

            if len(date) < 4:
                continue

            try:
                year = int(date[:4])
            except:
                continue

            if not (2017 <= year <= 2021):
                continue

            comment = row["CommentText"].strip()

            # remove empty comments
            if comment == "":
                continue

            # keep only English-like comments
            if not looks_english(comment):
                continue

            # remove duplicates
            if comment in seen:
                continue

            seen.add(comment)

            writer.writerow({
                "VideoID": row["VideoID"],
                "VideoTitle": row["VideoTitle"],
                "CommentText": comment,
                "PublishedAt": date
            })

            count += 1

            if count >= 1000:
                break

print("Done")