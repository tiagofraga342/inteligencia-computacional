import argparse
import os
import random

# Must be set before CUDA initializes. Reduces allocator fragmentation -- without it,
# a single near-512-token batch can OOM even though the average batch fits comfortably.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "checkpoints" / "baseline_qg"

MODEL_NAME = "t5-small"
# Answer-agnostic QG (no answer/topic given): baseline model from Fig. 2 of the paper,
# the "no topic" starting point that TopicQG (out of scope here) will later condition on.
INPUT_PREFIX = "generate question: "
ENCODER_MAX_LENGTH = 512  # matches the paper's encoder budget (section 3.3)
DECODER_MAX_LENGTH = 128  # matches the paper's decoder budget (section 3.3)
DEV_FRACTION = 0.05  # carved out of official train, for early stopping only
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=8000,
        help="Cap on official-train examples before the train/dev split (paper's 87599 takes "
        "~145 min/epoch on a 4GB GPU; pass -1 for the full set once the pipeline is validated).",
    )
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=4,
        help="Kept low on purpose: a batch that happens to land on several near-512-token "
        "SQuAD contexts OOMs this 4GB GPU well above this size (observed at 16).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=16,
        help="With the default batch size, 16 gives an effective batch of 64, matching the paper.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def split_by_article(dataset, dev_fraction, seed):
    # A row-level train_test_split would let the same SQuAD article (title) show up on
    # both sides, just with different questions about the same paragraph -- the model
    # could then look "better" on dev than it really generalizes. Splitting by title
    # instead keeps every article entirely on one side, matching how SQuAD's own
    # official train/validation split was made (442 vs. 48 disjoint articles).
    titles = sorted(set(dataset["title"]))
    rng = random.Random(seed)
    rng.shuffle(titles)
    n_dev_titles = max(1, round(dev_fraction * len(titles)))
    dev_titles = set(titles[:n_dev_titles])

    is_dev = [title in dev_titles for title in dataset["title"]]
    dev_indices = [i for i, flag in enumerate(is_dev) if flag]
    train_indices = [i for i, flag in enumerate(is_dev) if not flag]
    return dataset.select(train_indices), dataset.select(dev_indices)


def preprocess_batch(batch, tokenizer):
    inputs = [INPUT_PREFIX + c for c in batch["context"]]
    model_inputs = tokenizer(inputs, max_length=ENCODER_MAX_LENGTH, truncation=True)
    labels = tokenizer(text_target=batch["question"], max_length=DECODER_MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    args = parse_args()

    already_cached = CACHE_DIR.exists() and any("squad" in p.name.lower() for p in CACHE_DIR.glob("*"))
    if already_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"

    ds = load_dataset("rajpurkar/squad", cache_dir=str(CACHE_DIR))
    train_full = ds["train"]
    # Held out untouched: this becomes the reported test set (scripts/evaluate_qg.py),
    # the same split the paper treats as SQuAD's dev/test.
    test_split = ds["validation"]

    if args.max_train_samples >= 0 and args.max_train_samples < len(train_full):
        train_full = train_full.shuffle(seed=SEED).select(range(args.max_train_samples))

    train_split, dev_split = split_by_article(train_full, DEV_FRACTION, SEED)

    print(
        f"train={len(train_split)} ({len(set(train_split['title']))} articles)  "
        f"dev={len(dev_split)} ({len(set(dev_split['title']))} articles)  "
        f"test (SQuAD validation, held out)={len(test_split)}"
    )

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return preprocess_batch(batch, tokenizer)

    remove_cols = train_split.column_names
    train_tok = train_split.map(tokenize, batched=True, remove_columns=remove_cols)
    dev_tok = dev_split.map(tokenize, batched=True, remove_columns=remove_cols)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    effective_batch_size = args.per_device_batch_size * args.grad_accum_steps
    steps_per_epoch = max(1, -(-len(train_tok) // effective_batch_size))  # ceil div
    total_steps = steps_per_epoch * args.max_epochs
    warmup_steps = max(1, round(0.1 * total_steps))

    # Hyperparameters from the paper (section 3.3) where the GPU/time budget allows:
    # AdamW, lr=1e-3, eps=1e-8, effective batch size 64. Epoch count and training-set
    # size are reduced from the paper's 50 epochs / 87599 examples for compute-budget
    # reasons (benchmarked ~145 min/epoch on the full set on this GPU) -- a documented
    # deviation, not a reproduction error.
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=1e-3,
        adam_epsilon=1e-8,
        warmup_steps=warmup_steps,  # not in the paper; added after a smoke test showed fp16 instability without it
        num_train_epochs=args.max_epochs,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to=[],
        dataloader_num_workers=2,
        # Generation-based metrics are computed separately by evaluate_qg.py, so
        # training-time eval only needs loss -- cheaper and lighter on GPU memory.
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved best checkpoint (by dev loss) to {final_dir}")


if __name__ == "__main__":
    main()
