from sklearn.datasets import fetch_20newsgroups
import numpy as np

CATEGORIES = ['comp.graphics', 'sci.space', 'talk.religion.misc']

print("Loading dataset only once...")

newsgroups = fetch_20newsgroups(
    subset='train',
    categories=CATEGORIES,
    remove=('headers', 'footers', 'quotes')
)

print("Dataset loaded.")


def make_subset(n_samples):
    data = []
    labels = []

    for i in range(len(CATEGORIES)):
        idxs = np.where(newsgroups.target == i)[0][:n_samples]
        for idx in idxs:
            data.append(newsgroups.data[idx])
            labels.append(CATEGORIES[i])

    return data, labels
