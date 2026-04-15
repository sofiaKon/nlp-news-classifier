from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataset import make_subset


def load_model_1():
    print("Loading Model CountVectorizer...")

    data, labels = make_subset(20)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    print("Model CountVectorizer is ready.")
    return vectorizer, X, labels


def classify_model_1(text, vectorizer, X, labels):
    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, X)
    best_idx = np.argmax(sim)
    return labels[best_idx], sim[0][best_idx]
