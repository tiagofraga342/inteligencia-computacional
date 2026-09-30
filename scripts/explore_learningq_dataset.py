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
ALREADY_CACHED = CACHE_DIR.exists() and any("learningq" in p.name.lower() for p in CACHE_DIR.glob("*"))

DATASET_REPO = "sidovic/LearningQ-qg"  # cleaned mirror of Chen et al. 2018 (ICWSM), Khan Academy + TED-Ed
MODEL_NAME = "t5-small"
CONTEXT_MAX_TOKENS = 512  # same encoder budget used for the SQuAD baseline, for a fair comparison
QUESTION_MAX_TOKENS = 64

# `context` = full lecture transcript/article; `questionsrc` = the shorter excerpt the
# question was actually written about (LearningQ's closest analogue to a SQuAD answer span,
# except it's a passage, not a short span). See scripts/explore_squad_dataset.py for the
# SQuAD-side counterpart of every stat computed here.
QUESTION_TYPES = ["what", "who", "when", "where", "why", "how", "which", "whose", "is", "are", "do", "does", "did", "can", "was", "were"]


def classify_question_type(question):
    first_word = re.sub(r"[^a-zA-Z]", "", question.strip().split()[0]).lower() if question.strip() else ""
    return first_word if first_word in QUESTION_TYPES else "other"


def contains_why_how(question):
    q = question.lower()
    return bool(re.search(r"\bwhy\b", q) or re.search(r"\bhow\b", q))


ROW_BATCH_SIZE = 256


def token_lengths(tokenizer, texts):
    encoded = tokenizer(texts, return_attention_mask=False, verbose=False)["input_ids"]
    return [len(ids) for ids in encoded]


def main():
    if ALREADY_CACHED:
        print("LearningQ-qg already cached under data/cache, loading from disk (no download)...")
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("Downloading LearningQ-qg from Hugging Face Hub (will be cached under data/cache)...")

    ds = load_dataset(DATASET_REPO, cache_dir=str(CACHE_DIR))
    print(ds)

    train = ds["train"]
    n_train = len(train)
    n_val = len(ds["validation"])
    n_test = len(ds["test"])

    print(f"\nLoading {MODEL_NAME} tokenizer to compute real subword token lengths...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    # Streamed in small batches: whole columns of full transcripts don't fit in RAM.
    # Many questions share one transcript, so each context is measured once, keyed by hash.
    context_stats = {}
    context_lens, questionsrc_lens, question_lens = [], [], []
    context_token_lens, questionsrc_token_lens, question_token_lens = [], [], []
    question_types = Counter()
    why_how_count = 0

    for batch in train.iter(batch_size=ROW_BATCH_SIZE):
        new_contexts = {hash(c): c for c in batch["context"] if hash(c) not in context_stats}
        if new_contexts:
            texts = list(new_contexts.values())
            for key, text, n_tokens in zip(new_contexts, texts, token_lengths(tokenizer, texts)):
                context_stats[key] = (len(text.split()), n_tokens)
        for c in batch["context"]:
            n_words, n_tokens = context_stats[hash(c)]
            context_lens.append(n_words)
            context_token_lens.append(n_tokens)

        questionsrc_lens.extend(len(s.split()) for s in batch["questionsrc"])
        questionsrc_token_lens.extend(token_lengths(tokenizer, batch["questionsrc"]))
        question_lens.extend(len(q.split()) for q in batch["question"])
        question_token_lens.extend(token_lengths(tokenizer, batch["question"]))
        question_types.update(classify_question_type(q) for q in batch["question"])
        why_how_count += sum(contains_why_how(q) for q in batch["question"])

    print(f"{len(context_stats)} unique contexts across {n_train} rows")
    context_truncated = sum(1 for n in context_token_lens if n > CONTEXT_MAX_TOKENS)
    questionsrc_truncated = sum(1 for n in questionsrc_token_lens if n > CONTEXT_MAX_TOKENS)
    question_truncated = sum(1 for n in question_token_lens if n > QUESTION_MAX_TOKENS)
    why_how_fraction = why_how_count / n_train

    summary = {
        "dataset_repo": DATASET_REPO,
        "n_train_examples": n_train,
        "n_val_examples": n_val,
        "n_test_examples": n_test,
        "context_words_mean": sum(context_lens) / n_train,
        "context_words_min": min(context_lens),
        "context_words_max": max(context_lens),
        "questionsrc_words_mean": sum(questionsrc_lens) / n_train,
        "questionsrc_words_min": min(questionsrc_lens),
        "questionsrc_words_max": max(questionsrc_lens),
        "question_words_mean": sum(question_lens) / n_train,
        "question_words_min": min(question_lens),
        "question_words_max": max(question_lens),
        "context_tokens_mean": sum(context_token_lens) / n_train,
        f"context_pct_over_{CONTEXT_MAX_TOKENS}_tokens": 100 * context_truncated / n_train,
        "questionsrc_tokens_mean": sum(questionsrc_token_lens) / n_train,
        f"questionsrc_pct_over_{CONTEXT_MAX_TOKENS}_tokens": 100 * questionsrc_truncated / n_train,
        "question_tokens_mean": sum(question_token_lens) / n_train,
        f"question_pct_over_{QUESTION_MAX_TOKENS}_tokens": 100 * question_truncated / n_train,
        "question_type_counts": dict(question_types.most_common()),
        "why_how_anywhere_fraction": why_how_fraction,
    }

    print("\nSummary statistics (train split)")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(DATA_DIR / "learningq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExample rows")
    for i in range(3):
        ex = train[i]
        print(f"\n--- Example {i} ---")
        print("Context (truncated):", ex["context"][:300], "...")
        print("Question source excerpt (truncated):", ex["questionsrc"][:300], "...")
        print("Question:", ex["question"])

    # Figure 1: word-level length distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    axes1[0].hist(context_lens, bins=50, color="#4C72B0")
    axes1[0].set_title("Context length (words)")
    axes1[0].set_xlim(0, 3000)

    axes1[1].hist(questionsrc_lens, bins=50, color="#DD8452")
    axes1[1].set_title("Question-source excerpt length (words)")
    axes1[1].set_xlim(0, 400)

    axes1[2].hist(question_lens, bins=30, color="#55A868")
    axes1[2].set_title("Question length (words)")
    axes1[2].set_xlim(0, 60)

    plt.tight_layout()
    out_path_1 = FIG_DIR / "learningq_length_distributions.png"
    plt.savefig(out_path_1, dpi=150)
    print(f"\nSaved distribution plot to {out_path_1}")

    # Figure 2: question types and token-level truncation against the same 512/64 budget used for SQuAD
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))

    sorted_types = question_types.most_common()
    axes2[0, 0].bar([t for t, _ in sorted_types], [c for _, c in sorted_types], color="#8172B2")
    axes2[0, 0].set_title("Question type (first word)")
    axes2[0, 0].tick_params(axis="x", rotation=45)

    axes2[0, 1].hist(context_token_lens, bins=60, color="#4C72B0")
    axes2[0, 1].axvline(CONTEXT_MAX_TOKENS, color="black", linestyle="--", label=f"{CONTEXT_MAX_TOKENS} token limit")
    axes2[0, 1].set_title(f"Full context length (T5 tokens) — {100 * context_truncated / n_train:.1f}% truncated")
    axes2[0, 1].set_xlim(0, 6000)
    axes2[0, 1].legend()

    axes2[1, 0].hist(questionsrc_token_lens, bins=50, color="#C44E52")
    axes2[1, 0].axvline(CONTEXT_MAX_TOKENS, color="black", linestyle="--", label=f"{CONTEXT_MAX_TOKENS} token limit")
    axes2[1, 0].set_title(f"Question-source excerpt length (T5 tokens) — {100 * questionsrc_truncated / n_train:.1f}% truncated")
    axes2[1, 0].set_xlim(0, 1200)
    axes2[1, 0].legend()

    with open(DATA_DIR / "squad_summary.json") as f:
        squad_answer_words_mean = json.load(f)["answer_words_mean"]
    axes2[1, 1].bar(["SQuAD\n(answer span)", "LearningQ\n(questionsrc excerpt)"], [squad_answer_words_mean, summary["questionsrc_words_mean"]], color=["#64B5CD", "#C44E52"])
    axes2[1, 1].set_title("Answer/focus-span length (words): extractive vs. excerpt-based")

    plt.tight_layout()
    out_path_2 = FIG_DIR / "learningq_extra_distributions.png"
    plt.savefig(out_path_2, dpi=150)
    print(f"Saved extra distribution plot to {out_path_2}")


if __name__ == "__main__":
    main()
