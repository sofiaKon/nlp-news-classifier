"""
Main application for News Topic Classification.

This file:
- loads all models only once
- supports terminal mode
- supports Gradio web mode
"""

import gradio as gr
from model_Count import load_model_1, classify_model_1
from model_Tfidf import load_model_2, classify_model_2
from model_ngram import load_model_3, classify_model_3
from model_100 import load_model_4, classify_model_4

print("Loading all models...")

vectorizer_1, X_1, labels_1 = load_model_1()
vectorizer_2, X_2, labels_2 = load_model_2()
vectorizer_3, X_3, labels_3 = load_model_3()
vectorizer_4, X_4, labels_4 = load_model_4()

print("All models are loaded.")


def classify_all_models(text):
    """
    Classify input text using all 4 models.
    Returns a dictionary with model names, predicted labels, and similarity scores.
    """
    text = text.strip()

    label_1, score_1 = classify_model_1(text, vectorizer_1, X_1, labels_1)
    label_2, score_2 = classify_model_2(text, vectorizer_2, X_2, labels_2)
    label_3, score_3 = classify_model_3(text, vectorizer_3, X_3, labels_3)
    label_4, score_4 = classify_model_4(text, vectorizer_4, X_4, labels_4)

    return {
        "model_1": ("CountVectorizer (20 samples)", label_1, score_1),
        "model_2": ("TfidfVectorizer (20 samples)", label_2, score_2),
        "model_3": ("TF-IDF + ngram (20 samples)", label_3, score_3),
        "model_4": ("TfidfVectorizer (100 samples)", label_4, score_4),
    }


def format_results_table(text, results):
    """
    Format results for terminal output.
    """
    model_1_name, label_1, score_1 = results["model_1"]
    model_2_name, label_2, score_2 = results["model_2"]
    model_3_name, label_3, score_3 = results["model_3"]
    model_4_name, label_4, score_4 = results["model_4"]

    scores = [score_1, score_2, score_3, score_4]
    labels = [label_1, label_2, label_3, label_4]
    models = [model_1_name, model_2_name, model_3_name, model_4_name]

    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))

    lines = []
    lines.append("\n" + "=" * 110)
    lines.append(f"INPUT TEXT: {text}")
    lines.append("=" * 110)
    lines.append(f"{'Model':<40} | {'Category':<25} | {'Similarity':<10}")
    lines.append("-" * 110)
    lines.append(f"{model_1_name:<40} | {label_1:<25} | {score_1:.4f}")
    lines.append(f"{model_2_name:<40} | {label_2:<25} | {score_2:.4f}")
    lines.append(f"{model_3_name:<40} | {label_3:<25} | {score_3:.4f}")
    lines.append(f"{model_4_name:<40} | {label_4:<25} | {score_4:.4f}")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Detailed analysis:")
    lines.append(
        f"- Best model: {models[best_idx]} -> {labels[best_idx]} ({scores[best_idx]:.4f})")
    lines.append(
        f"- Lowest similarity: {models[worst_idx]} -> {labels[worst_idx]} ({scores[worst_idx]:.4f})")

    if len(set(labels)) == 1:
        lines.append("- All models predicted the same category.")
    else:
        lines.append("- The models predicted different categories.")

    if all(score == 0 for score in scores):
        lines.append("- No overlapping words were found in the training data.")

    lines.append("- CountVectorizer uses raw word frequency.")
    lines.append("- TfidfVectorizer reduces the effect of common words.")
    lines.append(
        "- ngram_range=(1,2) makes matching stricter by considering word pairs.")
    lines.append(
        "- Using 100 samples improves vocabulary coverage and often increases stability.")
    lines.append("=" * 110 + "\n")

    return "\n".join(lines)


def gradio_predict(text):
    """
    Format results for Gradio output.
    """
    text = text.strip()

    if not text:
        return "", "", "", ""

    results = classify_all_models(text)

    _, label_1, score_1 = results["model_1"]
    _, label_2, score_2 = results["model_2"]
    _, label_3, score_3 = results["model_3"]
    _, label_4, score_4 = results["model_4"]

    output_1 = f"{label_1} ({score_1:.4f})"
    output_2 = f"{label_2} ({score_2:.4f})"
    output_3 = f"{label_3} ({score_3:.4f})"
    output_4 = f"{label_4} ({score_4:.4f})"

    return output_1, output_2, output_3, output_4


def run_terminal():
    """
    Run terminal version.
    """
    print("\nTerminal mode is ready.")
    print("Type English text to classify.")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "exit":
            print("Program stopped.")
            break

        results = classify_all_models(text)
        print(format_results_table(text, results))


def run_gradio():
    """
    Run Gradio web interface.
    """
    demo = gr.Interface(
        fn=gradio_predict,
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

    demo.launch(share=True)


def main():
    print("\nChoose mode:")
    print("1. Terminal mode")
    print("2. Gradio web mode")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_terminal()
    elif choice == "2":
        run_gradio()
    else:
        print("Invalid choice. Please run the program again.")


if __name__ == "__main__":
    main()
