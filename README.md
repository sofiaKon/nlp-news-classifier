#  News Group Classification using Vectorization

## Project Description

This project implements a simple text classification system using vectorization techniques and cosine similarity.

The model classifies input text into one of three categories:

- comp.graphics
- sci.space
- talk.religion.misc

The classification is based on similarity between the input text and training documents.

---

##  Technologies Used

- Python
- scikit-learn
- NumPy

---

##  Methodology

The project uses:

- **CountVectorizer** – converts text into word frequency vectors
- **TfidfVectorizer** – reduces importance of common words
- **Cosine Similarity** – measures similarity between text vectors

The model does not use traditional machine learning training.  
Instead, it compares the input text with existing documents and selects the most similar one.

---
### 🧪 Q1. Why does similarity sometimes become 0.0000?

When a sentence such as:

"Exploring the mars with a robotic rover."

is entered into the model, the similarity score may become 0.0000 or the category may be incorrectly predicted.

This behavior is directly related to how **CountVectorizer** works.

---

### 🔹 How CountVectorizer Works

CountVectorizer converts text into a vector based on word frequency using a fixed vocabulary built from the training data.

Each word in the vocabulary becomes a feature (column), and each document is represented as a vector of word counts.

---

### 🔹 Reason 1: Vocabulary Mismatch

If the input sentence contains words that do not exist in the training data vocabulary, those words are ignored.

For example:

- "robotic"
- "rover"

If these words were not present in the training dataset, they will not appear in the vector at all.

As a result, the input vector may contain mostly zeros.

---

### 🔹 Reason 2: No Overlapping Words

Cosine similarity depends on overlapping words between vectors.

If the input text and training documents share no common words:

- dot product = 0
- similarity = 0.0000

---

### 🔹 Reason 3: Stop Words Removal

When using:

```python
CountVectorizer(stop_words='english')
```
---

##  Experiments (Q2)

Several experiments were conducted to improve the model:

### 1. CountVectorizer (20 samples)
Basic model using word counts.

### 2. TfidfVectorizer (20 samples)
Improves results by weighting important words higher.

### 3. TF-IDF with n-grams (1,2)
Includes word combinations (e.g., "space mission").

### 4. TF-IDF with 100 samples
Increasing dataset size improved performance and stability.

---

## 📊 Results

- TF-IDF produced more reliable results than CountVectorizer
- n-grams made the model more strict (lower similarity scores)
- Increasing samples from 20 to 100 improved similarity and stability

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the program:

```bash
python main.py
```

3. Enter English text:


```bash
rocket launch nasa orbit satellite
```

Example Output

```bash
Model                                | Category                | Similarity
-----------------------------------------------------------------------
CountVectorizer (20 samples)         | sci.space               | 0.1066
TfidfVectorizer (20 samples)         | sci.space               | 0.1014
TF-IDF + ngram (20 samples)          | sci.space               | 0.0704
TfidfVectorizer (100 samples)        | sci.space               | 0.1264
```

## Detailed Results Analysis

The similarity scores in this project may appear relatively low (e.g., 0.07–0.12), but this is expected for this type of text processing task.

### Why are the similarity scores low?

The model compares a short input sentence with full-length news documents.
Since the overlap of words is limited, the cosine similarity values tend to be small.

### Additionally:

The vector space is high-dimensional and sparse
Most words do not appear in every document
If words do not overlap → similarity approaches 0

Therefore, low values do not indicate poor performance.
The most important factor is whether the correct category is selected.

### Why does TF-IDF give lower scores than CountVectorizer?

TF-IDF reduces the importance of common words such as:

```bash
the, is, and, this
```

Instead, it emphasizes more informative words.

As a result:

- Similarity scores become lower
- But the model becomes more accurate and meaningful

### Why does n-gram produce the lowest scores?

When using:

```bash
ngram_range=(1, 2)
```

the model considers both:

- single words → "space"
- word pairs → "space mission"

This makes the model more strict because:

- both words in a phrase must match
- partial matches are less effective

Therefore, similarity scores decrease, but precision increases.

### Why does the 100-sample model give higher similarity?

Increasing the dataset size from 20 to 100 samples per category improves performance because:

- The vocabulary becomes richer
- There are more chances for word overlap
- The model has better coverage of each topic

As a result:

- Similarity scores increase
- Results become more stable
- Fewer cases of similarity = 0