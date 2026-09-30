import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EVAL_OUT_DIR = ROOT_DIR / "outputs" / "eval"

INPUT_PREFIX = "generate question: "  # must match scripts/train_baseline_qg.py
ENCODER_MAX_LENGTH = 512
GENERATION_MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT_DIR / "outputs" / "checkpoints" / "baseline_qg" / "final"),
        help="Path to a trained checkpoint (e.g. from scripts/train_baseline_qg.py).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=1000,
        help="Cap on the SQuAD validation split evaluated (-1 for the full 10570).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-examples-to-save", type=int, default=10)
    parser.add_argument("--output", default=str(EVAL_OUT_DIR / "baseline_squad_val.json"))
    return parser.parse_args()


def normalize_text(text):
    # Same normalization as the standard SQuAD evaluation script, applied here to
    # question strings instead of answer strings.
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_predictions(model, tokenizer, contexts, batch_size, num_beams):
    model.eval()
    predictions = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i : i + batch_size]
        inputs = tokenizer(
            [INPUT_PREFIX + c for c in batch],
            return_tensors="pt",
            truncation=True,
            max_length=ENCODER_MAX_LENGTH,
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=GENERATION_MAX_LENGTH, num_beams=num_beams)
        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(contexts))}/{len(contexts)}", end="\r")
    print()
    return predictions


def main():
    args = parse_args()
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    already_cached = CACHE_DIR.exists() and any("squad" in p.name.lower() for p in CACHE_DIR.glob("*"))
    if already_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"Loading checkpoint from {args.checkpoint} on {DEVICE}...")
    tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)

    # Untouched SQuAD validation split -- see scripts/train_baseline_qg.py for the split rationale.
    ds = load_dataset("rajpurkar/squad", cache_dir=str(CACHE_DIR))["validation"]
    if 0 <= args.max_eval_samples < len(ds):
        ds = ds.shuffle(seed=SEED).select(range(args.max_eval_samples))

    contexts, references = ds["context"], ds["question"]

    print(f"Generating predictions for {len(contexts)} examples (batch={args.batch_size}, beams={args.num_beams})...")
    predictions = generate_predictions(model, tokenizer, contexts, args.batch_size, args.num_beams)

    print("Computing metrics (BLEU, ROUGE, METEOR, F1)...")
    bleu = evaluate.load("sacrebleu").compute(predictions=predictions, references=[[r] for r in references])
    rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
    meteor = evaluate.load("meteor").compute(predictions=predictions, references=references)
    mean_f1 = sum(token_f1(p, r) for p, r in zip(predictions, references)) / len(predictions)

    metrics = {
        "checkpoint": args.checkpoint,
        "n_eval_examples": len(contexts),
        "bleu": bleu["score"],
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
        "meteor": float(meteor["meteor"]),
        "f1": mean_f1,
    }

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    examples = [
        {"context": c[:300], "reference_question": r, "generated_question": p}
        for c, r, p in list(zip(contexts, references, predictions))[: args.num_examples_to_save]
    ]

    with open(args.output, "w") as f:
        json.dump({"metrics": metrics, "examples": examples}, f, indent=2)

    print(f"\nSaved metrics + {len(examples)} example generations to {args.output}")


if __name__ == "__main__":
    main()
