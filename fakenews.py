import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# train_test_split: you have the full data set; you split it into 80% training data, 20% testing data
# the model is trained on 80% of the data and it is evaluated on the remaining 20% data, seeing how well it classifies news.

from sklearn.feature_extraction.text import TfidfVectorizer

# idea behind the vectorizer: we want the text and turn it into something and feed it into an ML model. We require something that
# can be represented using numbers.
# The vectorizer takes 2 metrics: TF-IDF (Term Frequency - Inverse Document Frequency)
# Term Frequnecy: no. of times a term appears in the document.
# IDF: a metric calculated with log and division. equals no. of documents / no. of documents that contain the term.
# You multiply the 2 metrics and get a score
# We vectorize the text using this metric

from sklearn.svm import LinearSVC

# for text data, a linear support vector classifier is powerful

data = pd.read_csv("fake_or_real_news.csv")
data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1)
print(data)
