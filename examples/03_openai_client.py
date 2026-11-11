"""Appel du serveur local (scripts/serve.py ou mlx_lm.server) avec le client OpenAI.

Terminal 1 :  uv run python scripts/serve.py mlx-community/gemma-4-e4b-it-4bit
Terminal 2 :  uv run python examples/03_openai_client.py
"""

import argparse

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
parser.add_argument("--prompt", default="Donne trois idées de tests à faire avec un LLM local.")
args = parser.parse_args()

# Le serveur MLX n'exige pas de clé, mais le client OpenAI en veut une non vide.
client = OpenAI(base_url=args.base_url, api_key="not-needed")

stream = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": args.prompt}],
    max_tokens=400,
    temperature=0.7,
    stream=True,
    # Paramètre propre au serveur MLX : coupe le mode « réflexion » (Gemma 4, Qwen 3…).
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
