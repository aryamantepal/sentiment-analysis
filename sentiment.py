# Sentiment Analysis for an article
from textblob import TextBlob
import nltk

nltk.download("punkt")

with open("mytext.txt", "r") as f:
    text = f.read()

blob = TextBlob(
    text
)  # passing the string, NOT the article object. The article object is just for getting a clean summary of the text

sentiment = blob.sentiment.polarity  # -1 to 1

print(sentiment)
