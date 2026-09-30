import argparse

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_NAME = "t5-small" # https://huggingface.co/google-t5/t5-small
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() # Sets the model into inference mode instead of training mode - https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval

    with open(args.input_path) as f:
        input_text = f.read().strip()

    tokenized_input = tokenizer(
        "summarize: " + input_text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    with torch.no_grad(): # Disables gradient tracking since we're not training - https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
        generated_ids = model.generate(**tokenized_input, max_length=64, num_beams=4)
    summary_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    with open(args.output_path, "w") as f:
        f.write(summary_output + "\n")

    print(f"Summary written to {args.output_path}")


if __name__ == "__main__":
    main()
