import sys
import time

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

CHECKPOINT = "outputs/checkpoints/baseline_qg_full_epoch3"
PREFIX = "generate question: "

EXAMPLES = {
    "1": (
        "Neural networks are trained using an algorithm called backpropagation, which "
        "calculates the gradient of the loss function with respect to each weight by "
        "applying the chain rule, propagating the error backward from the output layer."
    ),
    "2": (
        "The Amazon rainforest produces about 20% of the world's oxygen and is home to "
        "roughly 10% of all known species on Earth. Deforestation for agriculture and "
        "logging has reduced its original area by nearly 20% since 1970."
    ),
    "3": (
        "Vaccines work by training the immune system to recognize and fight pathogens. "
        "They introduce a harmless piece of a virus or bacteria, prompting the body to "
        "produce antibodies that provide protection against future infection."
    ),
}


def main():
    print("Carregando o modelo (checkpoint-357, CPU)...")
    t0 = time.time()
    tokenizer = T5TokenizerFast.from_pretrained(CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)
    model.eval()
    print(f"Pronto em {time.time() - t0:.1f}s.\n")

    while True:
        print("Escolha um contexto de exemplo, digite o seu próprio, ou 'sair':")
        for k, v in EXAMPLES.items():
            print(f"  [{k}] {v[:70]}...")
        choice = input("> ").strip()

        if choice.lower() in ("sair", "exit", "quit", "q"):
            break
        context = EXAMPLES.get(choice, choice)
        if not context:
            continue

        print(f"\nContexto: {context}\n")
        t0 = time.time()
        inputs = tokenizer(PREFIX + context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=64, num_beams=4)
        question = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        dt = time.time() - t0

        print(f"Pergunta gerada ({dt:.1f}s): {question}\n")
        print("-" * 70)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)
