# Servir un agent de code : maîtriser le prompt cache

Un agent de code (Claude Code, Aider, Cline…) appelle le serveur session après session. Chaque
requête renvoie tout le contexte : prompt système, définitions d'outils, historique. Sans cache,
le serveur relit tout à chaque tour (prefill) ; avec un cache sans plafond, la RAM se remplit.
Ce document règle les deux. Exemple chiffré : **MacBook M3, 24 Go, budget GPU 19,1 Go**.

## Comment `mlx_lm.server` cache les prompts

Le serveur garde en mémoire le KV cache des requêtes déjà traitées, dans un arbre de préfixes
(trie) géré en LRU (source : `mlx_lm/server.py` et `mlx_lm/models/cache.py`, version 0.32.2).

- Nouvelle requête : il cherche le plus long préfixe de tokens déjà calculé et ne fait le prefill
  que du reste. Le champ `usage.prompt_tokens_details.cached_tokens` de la réponse dit combien
  de tokens ont été réutilisés.
- Une copie du cache est faite à chaque réutilisation : le cache stocké n'est jamais modifié.
- Éviction : dès qu'un plafond est dépassé, le moins récemment utilisé part. Les caches sont
  rangés par type (`assistant`, `user`, `system`), et l'ordre d'éviction en tient compte.
- Si le cache d'un modèle est « rognable » (attention classique), une entrée plus longue sert
  aussi pour un préfixe plus court et les préfixes redondants sont supprimés. Les modèles
  hybrides (couches linéaires, convolution) ou à fenêtre glissante ne se rognent pas : seul un
  préfixe exactement enregistré est réutilisable.

### Les défauts sont dangereux

| Option | Défaut | Conséquence |
|---|---|---|
| `--prompt-cache-size` | 10 | Jusqu'à 10 caches distincts |
| `--prompt-cache-bytes` | **aucun plafond** | Seul le nombre d'entrées limite, pas leur taille |

Dix sessions de 32 000 tokens sur LFM2.5-24B pèsent 6,5 Go de cache. Il en reste 5,5 Go sous le
budget GPU une fois le modèle chargé : le Mac swappe. **Toujours fixer `--prompt-cache-bytes`.**

Format : nombre + `M`, `MB`, `G` ou `GB` (puissances de 10 : `2GB` = 2 000 000 000 octets).
Le plafond inclut les requêtes en cours de génération : le serveur retire leur taille du budget
avant de décider quoi évincer.

## Ce que coûte un token de contexte

KV cache mesuré en bf16 sur les modèles installés (`make_prompt_cache`, prefill de 1 à 8 K tokens) :

| Modèle | Poids | KV par token | Session de 32 K | Part fixe par cache |
|---|---|---|---|---|
| `LFM2.5-24B-A2B-MLX-4bit` | 13,6 Go | **20 Ko** | 0,65 Go | négligeable |
| `Qwen3.6-35B-A3B-oQ2.5e-mtp` | 14,9 Go | **20 Ko** | 0,65 Go | ~65 Mo (couches linéaires) |
| `gemma-4-e4b-it-4bit` | 5,1 Go | **56 Ko** | 1,8 Go | négligeable |

Les deux premiers n'ont que 10 couches d'attention complète sur 40 (le reste est à convolution
ou à attention linéaire), d'où un KV léger. Le calcul par formule de [ram.md](ram.md) redonne
20 Ko pour eux ; il sous-estime Gemma 4 (28 Ko calculés, 56 Ko mesurés) : mesure toujours.
La part fixe de Qwen vient de l'état des couches linéaires : un cache de 1 000 tokens fait déjà
~85 Mo. Le surcoût de Gemma s'explique par le fait que `make_prompt_cache` y alloue un cache
plein aussi sur les couches à fenêtre glissante (non vérifié pour le chemin du serveur).

## Budget sur le M3 24 Go

Budget GPU : 19,1 Go. À retirer : les poids, un pic de prefill (non mesuré ici ; compter
~1 Go de marge), puis le reste se partage entre la session active et le cache.

| Modèle | Reste après poids | Plafond conseillé | Capacité (tokens) | Sessions de 32 K |
|---|---|---|---|---|
| LFM2.5-24B (13,6 Go) | 5,5 Go | `--prompt-cache-bytes 2GB` | ~100 K | ~3 |
| Qwen3.6-35B 2 bits (14,9 Go) | 4,2 Go | `--prompt-cache-bytes 1500MB` | ~70 K | ~2 |
| Gemma 4 e4b (5,1 Go) | 14 Go | `--prompt-cache-bytes 6GB` | ~100 K | ~3 |

Ce sont des points de départ, pas des mesures de bout en bout : valide avec la section suivante.

```bash
uv run python scripts/serve.py LiquidAI/LFM2.5-24B-A2B-MLX-4bit -- \
  --prompt-cache-bytes 2GB --prompt-cache-size 6 --prefill-step-size 512 \
  --chat-template-args '{"enable_thinking":false}'
```

Le serveur affiche dans ses logs, après chaque requête : `Prompt Cache: N sequences, X GB`.
C'est le meilleur indicateur : X doit rester sous ton plafond.

## Réglages côté agent (aussi importants que ceux du serveur)

1. **Préfixe stable, au début.** Le cache ne sert que sur un préfixe identique token par token.
   Prompt système et outils d'abord, dans un ordre fixe, sans date, sans identifiant de session.
   Un seul token différent au début invalide tout ce qui suit.
2. **Historique en ajout seulement.** Si l'agent résume ou réécrit les anciens tours, le cache
   est perdu à partir du point modifié. Préférer compacter rarement et d'un coup.
3. **Bornes de contexte réalistes.** Sur 24 Go, vise 32 K de contexte pour l'agent
   (réglage de fenêtre de contexte de ton outil). Un 128 K plein sur LFM
   coûte 2,6 Go : un seul contexte avale tout le plafond conseillé.
4. **Une seule session active à la fois.** Les sous-agents en parallèle ajoutent chacun leur
   cache actif. Sur cette machine, limite à 1 ou 2.
5. **Désactiver la réflexion** (`enable_thinking: false`) : moins de tokens générés, donc moins
   de KV, et un historique plus court à rejouer.

## Quand le cache ne suffit plus

| Symptôme | Cause probable | Action |
|---|---|---|
| Latence du premier token qui remonte à chaque tour | Préfixe instable ou cache évincé | Regarder `cached_tokens` ; stabiliser le prompt ; monter le plafond |
| Pression mémoire jaune/rouge, swap | Plafond trop haut, ou contexte trop grand | Baisser `--prompt-cache-bytes`, borner le contexte de l'agent |
| `--kv-bits` refusé | Modèle à fenêtre glissante (Gemma) | Pas de quantification du KV sur ces modèles |
| Requêtes servies une par une | `--kv-bits` désactive le batching | Ne l'utiliser que si le gain mémoire est nécessaire |

`--kv-bits 8` divise le KV par ~2 mais passe le serveur en traitement séquentiel. Pour un
agent unique, c'est acceptable ; pour plusieurs, préfère un plafond de cache plus bas.

## Mesurer sur ta machine

`examples/04_prompt_cache_agent.py` simule un agent : un prompt système commun, plusieurs
sessions, des tours qui s'ajoutent. Il affiche pour chaque requête les tokens du prompt, ceux
servis par le cache, et la latence.

```bash
# Terminal 1
uv run python scripts/serve.py LiquidAI/LFM2.5-24B-A2B-MLX-4bit -- --prompt-cache-bytes 2GB
# Terminal 2
uv run python examples/04_prompt_cache_agent.py --model LiquidAI/LFM2.5-24B-A2B-MLX-4bit
# Terminal 3 : pression mémoire pendant le test
memory_pressure | grep "free percentage"
```

À lire dans le résultat : premier tour d'une session nouvelle = `cached` bas (seul le prompt
système est partagé) ; tours suivants = `cached` proche de `prompt` et latence bien plus basse.
Si `cached` reste à 0 sur les tours suivants, le préfixe n'est pas stable ou le cache est évincé.

### Résultat mesuré (Gemma 4 e4b, `--prompt-cache-bytes 1GB`, `--sessions 3 --turns 3 --revisit`)

```text
session tours   prompt   cached     hit    latence
      1     0     3712        0     0 %      8.5 s      ← tout est calculé
      1     2     8108     5909    73 %      4.8 s      ← seul le dernier tour est calculé
      2     0     3712     3698  100 %      0.4 s      ← prompt système réutilisé d'une session à l'autre
--- retour sur la session 1 ---
      1     2     8108     3698    46 %      9.6 s      ← son cache a été évincé, il faut recalculer
```

Dans les logs : `Prompt Cache: 9 sequences, 1.18 GB`. Le plafond de 1 Go est donc tenu à ~0,2 Go
près (le serveur déduit les requêtes actives, puis évince). Sur ce modèle chaque requête laisse
une entrée (9 requêtes, 9 séquences) : Gemma ne se rogne pas, d'où l'importance du plafond.
Le test n'a pas été fait avec LFM2.5-24B ni Qwen3.6 : les plafonds du tableau ci-dessus restent
à valider avec ce même script.
