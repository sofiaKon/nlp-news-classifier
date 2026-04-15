from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CATEGORIES = ['comp.graphics', 'sci.space', 'talk.religion.misc']


def load_model_1():
    print("Loading Model CountVectorizer...")

    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes')
    )

    data = []
    labels = []

    for i in range(len(CATEGORIES)):
        idxs = np.where(newsgroups.target == i)[0][:20]
        for idx in idxs:
            data.append(newsgroups.data[idx])
            labels.append(CATEGORIES[i])

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    print("Model CountVectorizer is ready.")
    return vectorizer, X, labels


def classify_model_1(text, vectorizer, X, labels):
    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, X)
    best_idx = np.argmax(sim)
    return labels[best_idx], sim[0][best_idx]
