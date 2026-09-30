import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "outputs" / "eval"
DATASET_REPO = "sidovic/LearningQ-qg"

# Keyword heuristic: LearningQ-qg exposes no course/subject column that we know of, so
# "computing" is approximated from the text. Terms are matched as whole words (a prefix
# match for stems ending in "*") and kept specific to avoid e.g. "function" from math.
STRONG = [
    "algorithm*", "program", "programming", "programmer*", "javascript", "python", "html", "css",
    "sql", "compiler", "binary", "bit", "bits", "byte*", "cpu", "processor*", "ram", "memory",
    "software", "hardware", "database*", "array*", "loop*", "variable*", "recursion", "recursive",
    "boolean", "string*", "debug*", "code", "coding", "internet", "network*", "encrypt*",
    "cryptograph*", "computer*", "pixel*", "bandwidth", "packet*", "server*", "browser*",
]
PATTERN = re.compile(
    r"\b(?:" + "|".join(t[:-1] + r"\w*" if t.endswith("*") else re.escape(t) for t in STRONG) + r")\b",
    re.IGNORECASE,
)
TEXT_FIELDS = ("question", "questionsrc")


def is_computing(row, min_hits):
    text = " ".join(str(row.get(f, "")) for f in TEXT_FIELDS)
    return len(set(m.lower() for m in PATTERN.findall(text))) >= min_hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-hits", type=int, default=2, help="distinct keywords required in question+questionsrc")
    ap.add_argument("--sample", type=int, default=50, help="rows written for manual review")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(DATASET_REPO, cache_dir=str(CACHE_DIR))
    print(ds)
    print("columns:", {s: ds[s].column_names for s in ds})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = {}
    for split in ds:
        rows = [r for r in ds[split] if is_computing(r, args.min_hits)]
        kept[split] = rows
        print(f"{split}: {len(rows)} / {len(ds[split])} ({100 * len(rows) / max(1, len(ds[split])):.1f}%)")

    with open(OUT_DIR / "learningq_computing.jsonl", "w") as f:
        for split, rows in kept.items():
            for r in rows:
                f.write(json.dumps({"split": split, **{k: r[k] for k in TEXT_FIELDS if k in r}}, ensure_ascii=False) + "\n")

    allrows = [(s, r) for s, rows in kept.items() for r in rows]
    random.Random(args.seed).shuffle(allrows)
    with open(OUT_DIR / "learningq_computing_sample.txt", "w") as f:
        for i, (s, r) in enumerate(allrows[: args.sample], 1):
            f.write(f"#{i} [{s}]\nQ: {r.get('question')}\nSRC: {str(r.get('questionsrc'))[:400]}\n\n")
    print(f"wrote {OUT_DIR}/learningq_computing.jsonl and learningq_computing_sample.txt (review the sample by hand)")


if __name__ == "__main__":
    main()
