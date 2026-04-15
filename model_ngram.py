from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataset import make_subset


def load_model_3():
    print("Loading Model TF-IDF + ngram...")

    data, labels = make_subset(20)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(data)

    print("Model TF-IDF + ngram is ready.")
    return vectorizer, X, labels


def classify_model_3(text, vectorizer, X, labels):
    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, X)
    best_idx = np.argmax(sim)
    return labels[best_idx], sim[0][best_idx]
