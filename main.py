# Sentiment Analysis for an article from the internet
from textblob import TextBlob
from newspaper import Article
import nltk

nltk.download("punkt")

# Getting the article into the script to perform a sentiment analysis on to it.
# Scores range from -1 (hateful, negative) to 1 (happy, positive). Objective articles would have a neutral score.
url = "https://en.wikipedia.org/wiki/Mathematics"

# Transforming it into an article object of the newspaper library:
article = Article(url)  # this is the object we use to get the article into the script

article.download()  # gets the article into the script
article.parse()  # makes it readable / gets all the HTML out of it
article.nlp()  # preparing it for NLP (Natural Language Processing)

text = article.summary
# gets the ENTIRE text of the article. article.summary gets the summary - it focuses more on what's
# actually important in the text
print(text)

blob = TextBlob(
    text
)  # passing the string, NOT the article object. The article object is just for getting a clean summary of the text

sentiment = blob.sentiment.polarity  # -1 to 1

print(sentiment)
