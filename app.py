"""
Gradio Web Application for News Topic Classification

This application allows users to input English text and compare predictions
from four different vectorization models:

1. CountVectorizer (20 samples)
2. TF-IDF (20 samples)
3. TF-IDF with n-grams (1,2)
4. TF-IDF (100 samples)

Each model predicts:
- the most similar category
- cosine similarity score

Purpose:
- demonstrate how vectorization affects similarity
- compare model performance
- show impact of dataset size and feature engineering
"""
import gradio as gr
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORIES = ['comp.graphics', 'sci.space', 'talk.religion.misc']


def load_data(n_samples):
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes')
    )

    data = []
    labels = []

    for i in range(len(CATEGORIES)):
        idxs = np.where(newsgroups.target == i)[0][:n_samples]
        for idx in idxs:
            data.append(newsgroups.data[idx])
            labels.append(CATEGORIES[i])

    return data, labels


print("Loading models...")

# Model 1: Count (20)
data1, labels1 = load_data(20)
vec1 = CountVectorizer(stop_words='english')
X1 = vec1.fit_transform(data1)

# Model 2: TF-IDF (20)
data2, labels2 = load_data(20)
vec2 = TfidfVectorizer(stop_words='english')
X2 = vec2.fit_transform(data2)

# Model 3: TF-IDF + ngram
data3, labels3 = load_data(20)
vec3 = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X3 = vec3.fit_transform(data3)

# Model 4: TF-IDF (100)
data4, labels4 = load_data(100)
vec4 = TfidfVectorizer(stop_words='english')
X4 = vec4.fit_transform(data4)

print("All models loaded.")


def classify_all(text):
    text = text.strip()

    if not text:
        return "Enter text", "", "", ""

    def predict(vec, X, labels):
        v = vec.transform([text])
        sim = cosine_similarity(v, X)
        best = np.argmax(sim)
        return f"{labels[best]} ({sim[0][best]:.4f})"

    r1 = predict(vec1, X1, labels1)
    r2 = predict(vec2, X2, labels2)
    r3 = predict(vec3, X3, labels3)
    r4 = predict(vec4, X4, labels4)

    return r1, r2, r3, r4


demo = gr.Interface(
    fn=classify_all,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter English text...",
        label="Input Text"
    ),
    outputs=[
        gr.Textbox(label="CountVectorizer (20 samples)"),
        gr.Textbox(label="TF-IDF (20 samples)"),
        gr.Textbox(label="TF-IDF + ngram (1,2)"),
        gr.Textbox(label="TF-IDF (100 samples)")
    ],
    title="News Topic Classification (Comparison)",
    description="Compare 4 models: CountVectorizer, TF-IDF, n-gram, and larger dataset."
)

if __name__ == "__main__":
    demo.launch(share=True)
