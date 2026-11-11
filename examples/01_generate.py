"""Génération simple avec l'API Python de mlx_lm.

uv run python examples/01_generate.py
uv run python examples/01_generate.py --model <repo> --prompt "Ta question"
"""

import argparse

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
parser.add_argument("--prompt", default="Explique en trois phrases ce qu'est la quantification d'un LLM.")
parser.add_argument("--max-tokens", type=int, default=300)
args = parser.parse_args()

# Charge les poids (téléchargés dans ~/.cache/huggingface/hub au premier appel).
model, tokenizer = load(args.model)

# Les modèles instruct attendent leur chat template ; enable_thinking=False coupe
# le mode « réflexion » des modèles qui en ont un (Gemma 4, Qwen 3…), ignoré sinon.
messages = [{"role": "user", "content": args.prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

text = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=args.max_tokens,
    sampler=make_sampler(temp=0.7, top_p=0.9),
    verbose=True,  # affiche le texte puis les stats : tokens/s et pic mémoire
)
