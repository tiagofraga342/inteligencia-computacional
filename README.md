# Inteligência Computacional

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate sentencepiece evaluate rouge-score sacrebleu nltk pandas matplotlib
```

## Scripts

```bash
source .venv/bin/activate

python scripts/explore_pretrained_t5.py examples/input.txt examples/output.txt

python scripts/explore_squad_dataset.py
python scripts/explore_learningq_dataset.py

# Fine-tunes T5-small on SQuAD for answer-agnostic QG (no topic yet -- this is the
# "Baseline" model from Fig. 2 of the paper). --max-train-samples -1 uses the full
# official train split (87599 examples); the default (8000) is a compute-budget
# reduction, see note below.
python scripts/train_baseline_qg.py

# Generates questions with a trained checkpoint on SQuAD validation (held out from
# training) and reports BLEU/ROUGE/METEOR/F1.
python scripts/evaluate_qg.py --checkpoint outputs/checkpoints/baseline_qg/final
```

### Deviations from the paper's hyperparameters (section 3.3)

The paper fine-tunes with batch size 64, sequence lengths 512/128, up to 50 epochs, on
the full ~87.6k-example train split. Benchmarked on this project's GPU (GTX 1650, 4GB),
that recipe is ~145 minutes/epoch on the full training set -- too slow for the
available time budget. `train_baseline_qg.py` keeps the optimizer, learning rate,
epsilon and effective batch size (64, via gradient accumulation) from the paper, but
defaults to a smaller training subset and fewer epochs (both configurable via CLI
flags) with early stopping on dev loss. This is a documented compute-budget reduction,
not a reproduction error -- see `--max-train-samples -1` above to run the full recipe
given enough time.
