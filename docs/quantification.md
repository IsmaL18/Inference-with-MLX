# Quantification : rappels rapides

Quantifier, c'est stocker chaque poids sur moins de bits que les 16 du float16. Le modèle
est plus petit, plus rapide (moins d'octets à lire par token) et un peu moins bon.

## Les formats courants

| Format | Bits effectifs | Taille vs float16 | Qualité | Quand l'utiliser |
|---|---|---|---|---|
| `bf16` / `fp16` | 16 | 100 % | Référence | Petits modèles, conversion, fine-tuning |
| `8bit` (affine) | ~8,5 | ~53 % | Quasi identique | Quand la RAM le permet |
| `6bit` | ~6,5 | ~40 % | Très proche | Bon compromis |
| `4bit` (affine) | ~4,5 | ~28 % | Légère perte, souvent imperceptible en chat | **Le standard sur Mac** |
| `3bit` | ~3,5 | ~22 % | Perte visible (raisonnement, code, précision factuelle) | Faire tenir un gros modèle |
| `2bit` | ~2,5 | ~16 % | Forte dégradation | Rarement utile |

« Affine » (le mode par défaut de MLX) : les poids sont découpés en groupes de 64
(`--q-group-size`), chaque groupe a son échelle et son biais. D'où le ~0,5 bit de plus par poids.

## Les autres modes de MLX

| Mode | Principe |
|---|---|
| `mxfp4` | Flottant 4 bits au standard *microscaling* (OCP MX) : groupes de 32, une échelle partagée |
| `nvfp4` | Variante 4 bits de NVIDIA, groupes de 16 |
| `mxfp8` | Microscaling en 8 bits |
| `mixed_3_6`, `mixed_4_6`… | Bits mixtes : les couches les plus sensibles gardent plus de bits |

## Quantification « intelligente »

Le `-q` de `mlx_lm.convert` arrondit chaque poids sans regarder les données. Ces méthodes
utilisent un petit jeu de calibration pour perdre moins de qualité, au même nombre de bits :

| Commande | Méthode |
|---|---|
| `mlx_lm.awq` | AWQ : protège les poids qui comptent le plus pour les activations |
| `mlx_lm.gptq` | GPTQ : compense l'erreur d'arrondi couche par couche |
| `mlx_lm.dwq` | DWQ : distille le modèle complet vers sa version quantifiée |
| `mlx_lm.dynamic_quant` | Choisit les bits par couche selon leur sensibilité |

## Lire un nom de modèle

`unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` :

| Morceau | Sens |
|---|---|
| `unsloth/` | L'auteur de la conversion (`mlx-community` = la communauté MLX de référence) |
| `Qwen3.6` | La famille et la version |
| `35B` | 35 milliards de paramètres au total |
| `A3B` | MoE : 3 milliards de paramètres *actifs* par token |
| `UD` | « Unsloth Dynamic » : quantification mixte d'Unsloth |
| `MLX` | Format MLX |
| `3bit` | Quantification |

Autres suffixes fréquents : `it` / `Instruct` (affiné pour suivre des instructions : ce qu'il
te faut pour du chat), `VL` (vision-langage), `e4b` (Gemma : ~4B paramètres *effectifs*).
Les conventions varient d'un auteur à l'autre : en cas de doute, la clé `quantization` du
`config.json` fait foi (`bits`, `group_size`, `mode`).

## Quantifier le KV cache

Indépendant des poids : `--kv-bits 8` ou `4` réduit la mémoire du contexte, pas celle du
modèle. Voir [contexte-performance.md](contexte-performance.md).

## Règle pratique

1. Prends le plus gros modèle qui tient en **4bit** dans ton budget (voir [ram.md](ram.md)).
2. S'il reste de la marge, passe en 6 ou 8bit.
3. Évite le 3bit et en dessous, sauf pour faire tenir un modèle nettement plus gros : un 32B
   en 3bit bat souvent un 14B en 8bit.
