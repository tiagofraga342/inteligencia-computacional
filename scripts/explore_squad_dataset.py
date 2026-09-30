import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import T5TokenizerFast

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / "cache"
ALREADY_CACHED = (CACHE_DIR / "rajpurkar___squad").exists()

MODEL_NAME = "t5-small"  # https://huggingface.co/google-t5/t5-small
CONTEXT_MAX_TOKENS = 512  # matches the encoder max_length used for fine-tuning
QUESTION_MAX_TOKENS = 64  # matches the decoder max_length used for generation

QUESTION_TYPES = ["what", "who", "when", "where", "why", "how", "which", "whose", "is", "are", "do", "does", "did", "can", "was", "were"]


def classify_question_type(question):
    first_word = re.sub(r"[^a-zA-Z]", "", question.strip().split()[0]).lower() if question.strip() else ""
    return first_word if first_word in QUESTION_TYPES else "other"


def contains_why_how(question):
    q = question.lower()
    return bool(re.search(r"\bwhy\b", q) or re.search(r"\bhow\b", q))


def main():
    if ALREADY_CACHED:
        print("SQuAD 1.1 already cached under data/cache, loading from disk (no download)...")
        # https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("Downloading SQuAD 1.1 from Hugging Face Hub (will be cached under data/cache)...")

    ds = load_dataset("rajpurkar/squad", cache_dir=str(CACHE_DIR)) # https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset
    print(ds)

    train, val = ds["train"], ds["validation"]

    n_train, n_val = len(train), len(val)
    train_titles = set(train["title"])
    val_titles = set(val["title"])

    context_lens = [len(c.split()) for c in train["context"]]
    question_lens = [len(q.split()) for q in train["question"]]
    answer_lens = [len(a["text"][0].split()) if a["text"] else 0 for a in train["answers"]]

    print(f"\nLoading {MODEL_NAME} tokenizer to compute real subword token lengths...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    context_token_lens = [len(ids) for ids in tokenizer(list(train["context"]))["input_ids"]]
    question_token_lens = [len(ids) for ids in tokenizer(list(train["question"]))["input_ids"]]
    context_truncated = sum(1 for n in context_token_lens if n > CONTEXT_MAX_TOKENS)
    question_truncated = sum(1 for n in question_token_lens if n > QUESTION_MAX_TOKENS)

    question_types = Counter(classify_question_type(q) for q in train["question"])
    why_how_fraction = sum(contains_why_how(q) for q in train["question"]) / n_train

    context_question_counts = Counter(train["context"])
    title_question_counts = Counter(train["title"])
    top_titles = title_question_counts.most_common(15)

    summary = {
        "n_train_examples": n_train,
        "n_val_examples": n_val,
        "n_unique_articles_train": len(train_titles),
        "n_unique_articles_val": len(val_titles),
        "context_words_mean": sum(context_lens) / n_train,
        "context_words_min": min(context_lens),
        "context_words_max": max(context_lens),
        "question_words_mean": sum(question_lens) / n_train,
        "question_words_min": min(question_lens),
        "question_words_max": max(question_lens),
        "answer_words_mean": sum(answer_lens) / n_train,
        "context_tokens_mean": sum(context_token_lens) / n_train,
        f"context_pct_over_{CONTEXT_MAX_TOKENS}_tokens": 100 * context_truncated / n_train,
        "question_tokens_mean": sum(question_token_lens) / n_train,
        f"question_pct_over_{QUESTION_MAX_TOKENS}_tokens": 100 * question_truncated / n_train,
        "n_unique_contexts_train": len(context_question_counts),
        "questions_per_context_mean": n_train / len(context_question_counts),
        "question_type_counts": dict(question_types.most_common()),
        "why_how_anywhere_fraction": why_how_fraction,
    }

    print("\nSummary statistics (train split)")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(DATA_DIR / "squad_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExample rows")
    for i in range(3):
        ex = train[i]
        print(f"\n--- Example {i} (article: {ex['title']}) ---")
        print("Context:", ex["context"][:300], "...")
        print("Question:", ex["question"])
        print("Answer:", ex["answers"]["text"][0])

    # Figure 1: word-level length distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    axes1[0].hist(context_lens, bins=50, color="#4C72B0")
    axes1[0].set_title("Context length (words)")
    axes1[0].set_xlim(0, 400)

    axes1[1].hist(question_lens, bins=30, color="#DD8452")
    axes1[1].set_title("Question length (words)")
    axes1[1].set_xlim(0, 40)

    axes1[2].hist(answer_lens, bins=20, color="#55A868")
    axes1[2].set_title("Answer length (words)")
    axes1[2].set_xlim(0, 20)

    plt.tight_layout()
    out_path_1 = FIG_DIR / "squad_length_distributions.png"
    plt.savefig(out_path_1, dpi=150)
    print(f"\nSaved distribution plot to {out_path_1}")

    # Figure 2: question types, token lengths, per-context question counts, top articles
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))

    sorted_types = question_types.most_common()
    axes2[0, 0].bar([t for t, _ in sorted_types], [c for _, c in sorted_types], color="#8172B2")
    axes2[0, 0].set_title("Question type (first word)")
    axes2[0, 0].tick_params(axis="x", rotation=45)

    axes2[0, 1].hist(context_token_lens, bins=50, color="#4C72B0")
    axes2[0, 1].axvline(CONTEXT_MAX_TOKENS, color="black", linestyle="--", label=f"{CONTEXT_MAX_TOKENS} token limit")
    axes2[0, 1].set_title(f"Context length (T5 tokens) — {100 * context_truncated / n_train:.1f}% truncated")
    axes2[0, 1].set_xlim(0, 700)
    axes2[0, 1].legend()

    counts_per_context = list(context_question_counts.values())
    axes2[1, 0].hist(counts_per_context, bins=range(1, max(counts_per_context) + 2), color="#C44E52", align="left")
    axes2[1, 0].set_title("Questions per context paragraph")

    top_titles_reversed = list(reversed(top_titles))
    axes2[1, 1].barh([t for t, _ in top_titles_reversed], [c for _, c in top_titles_reversed], color="#64B5CD")
    axes2[1, 1].set_title("Top 15 articles by question count")

    plt.tight_layout()
    out_path_2 = FIG_DIR / "squad_extra_distributions.png"
    plt.savefig(out_path_2, dpi=150)
    print(f"Saved extra distribution plot to {out_path_2}")


if __name__ == "__main__":
    main()
