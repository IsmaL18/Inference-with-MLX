"""Génération en streaming, token par token, avec les métriques de performance.

uv run python examples/02_stream.py
"""

import argparse

from mlx_lm import load, stream_generate

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
parser.add_argument("--prompt", default="Écris un haïku sur un Mac qui fait tourner un LLM.")
parser.add_argument("--max-tokens", type=int, default=200)
args = parser.parse_args()

model, tokenizer = load(args.model)
messages = [{"role": "user", "content": args.prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

response = None
for response in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
    print(response.text, end="", flush=True)

if response:
    print("\n")
    print(f"Prompt     : {response.prompt_tokens} tokens à {response.prompt_tps:.1f} tokens/s (prefill)")
    print(f"Génération : {response.generation_tokens} tokens à {response.generation_tps:.1f} tokens/s (decode)")
    print(f"Pic mémoire: {response.peak_memory:.2f} Go")
