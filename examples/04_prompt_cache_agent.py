"""Simule un agent de code pour mesurer le prompt cache du serveur.

Terminal 1 :  uv run python scripts/serve.py <modèle> -- --prompt-cache-bytes 2GB
Terminal 2 :  uv run python examples/04_prompt_cache_agent.py --model <modèle>

Chaque session partage le même prompt système (comme un agent réel), puis ajoute des tours.
Pour chaque requête : tokens du prompt, tokens servis par le cache (`cached_tokens`), latence.
"""

import argparse
import time

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
parser.add_argument("--sessions", type=int, default=4, help="nombre de sessions successives")
parser.add_argument("--turns", type=int, default=3, help="tours par session")
parser.add_argument("--system-rules", type=int, default=150, help="taille du prompt système (nb de règles)")
parser.add_argument("--turn-lines", type=int, default=60, help="taille de chaque sortie d'outil simulée")
parser.add_argument("--revisit", action="store_true", help="repasse sur la session 1 à la fin (a-t-elle été évincée ?)")
args = parser.parse_args()

client = OpenAI(base_url=args.base_url, api_key="not-needed")

SYSTEM = "Tu es un agent de code. Règles :\n" + "\n".join(
    f"{i}. Respecte la convention numéro {i} du dépôt et ne modifie que les fichiers demandés."
    for i in range(args.system_rules)
)


def tool_output(session: int, turn: int) -> str:
    return "\n".join(
        f"src/module_{session}/fichier_{turn}_{i}.py:{i}: def fonction_{session}_{turn}_{i}(x): return x + {i}"
        for i in range(args.turn_lines)
    )


def history(session: int, turns: int) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Session {session} : relis le code."},
    ]
    for turn in range(turns):
        messages.append({"role": "assistant", "content": f"Je lis les fichiers du tour {turn}."})
        messages.append({"role": "user", "content": tool_output(session, turn)})
    return messages


def call(session: int, turns: int) -> None:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=history(session, turns),
        max_tokens=8,  # on mesure le prefill, pas la génération
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.perf_counter() - start
    prompt = response.usage.prompt_tokens
    details = getattr(response.usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) if details else 0) or 0
    print(f"{session:>7} {turns:>5} {prompt:>8} {cached:>8} {100 * cached / prompt:>5.0f} % {elapsed:>8.1f} s")


print(f"{'session':>7} {'tours':>5} {'prompt':>8} {'cached':>8} {'hit':>7} {'latence':>10}")
for session in range(1, args.sessions + 1):
    for turns in range(args.turns):
        call(session, turns)
if args.revisit:
    print("--- retour sur la session 1 ---")
    call(1, args.turns - 1)
