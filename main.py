from model_Count import load_model_1, classify_model_1
from model_Tfidf import load_model_2, classify_model_2
from model_ngram import load_model_3, classify_model_3
from model_100 import load_model_4, classify_model_4


def print_table(
    text,
    label_1, score_1,
    label_2, score_2,
    label_3, score_3,
    label_4, score_4
):
    print("\n" + "=" * 110)
    print(f"INPUT TEXT: {text}")
    print("=" * 110)
    print(f"{'Model':<40} | {'Category':<25} | {'Similarity':<10}")
    print("-" * 110)
    print(f"{'CountVectorizer (20 samples)':<40} | {label_1:<25} | {score_1:.4f}")
    print(f"{'TfidfVectorizer (20 samples)':<40} | {label_2:<25} | {score_2:.4f}")
    print(f"{'TF-IDF + ngram (20 samples)':<40} | {label_3:<25} | {score_3:.4f}")
    print(f"{'TfidfVectorizer (100 samples)':<40} | {label_4:<25} | {score_4:.4f}")
    print("=" * 110)

    scores = [score_1, score_2, score_3, score_4]
    labels = [label_1, label_2, label_3, label_4]
    models = [
        "CountVectorizer (20 samples)",
        "TfidfVectorizer (20 samples)",
        "TF-IDF + ngram (20 samples)",
        "TfidfVectorizer (100 samples)"
    ]

    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))

    print("\nDetailed analysis:")
    print(
        f"- Best model: {models[best_idx]} -> {labels[best_idx]} ({scores[best_idx]:.4f})")
    print(
        f"- Lowest similarity: {models[worst_idx]} -> {labels[worst_idx]} ({scores[worst_idx]:.4f})")

    if len(set(labels)) == 1:
        print("- All models predicted the same category.")
    else:
        print("- The models predicted different categories.")

    if all(score == 0 for score in scores):
        print("- No overlapping words were found in the training data.")

    print("- CountVectorizer uses raw word frequency.")
    print("- TfidfVectorizer reduces the effect of common words.")
    print("- ngram_range=(1,2) makes matching stricter by considering word pairs.")
    print("- Using 100 samples improves vocabulary coverage and often increases stability.")
    print("=" * 110 + "\n")


def main():
    vectorizer_1, X_1, labels_1 = load_model_1()
    vectorizer_2, X_2, labels_2 = load_model_2()
    vectorizer_3, X_3, labels_3 = load_model_3()
    vectorizer_4, X_4, labels_4 = load_model_4()

    print("\nAll models are loaded.")
    print("Type English text to classify.")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "exit":
            print("Program stopped.")
            break

        label_1, score_1 = classify_model_1(text, vectorizer_1, X_1, labels_1)
        label_2, score_2 = classify_model_2(text, vectorizer_2, X_2, labels_2)
        label_3, score_3 = classify_model_3(text, vectorizer_3, X_3, labels_3)
        label_4, score_4 = classify_model_4(text, vectorizer_4, X_4, labels_4)

        print_table(
            text,
            label_1, score_1,
            label_2, score_2,
            label_3, score_3,
            label_4, score_4
        )


if __name__ == "__main__":
    main()
