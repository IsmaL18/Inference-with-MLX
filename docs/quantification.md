# Quantification : rappels rapides

Quantifier, c'est stocker chaque poids sur moins de bits que les 16 du bf16/float16. Le modèle
est plus petit, plus rapide (moins d'octets à lire par token) et un peu moins bon.

## Les formats, du plus précis au plus compressé

| Format | Bits effectifs | Taille vs bf16 | Qualité | MLX | Quand l'utiliser |
|---|---|---|---|---|---|
| `fp32` | 32 | 200 % | Référence absolue | Oui | Jamais pour l'inférence : 2× plus lourd pour rien |
| `bf16` / `fp16` | 16 | 100 % | Référence | Oui | Petits modèles, conversion, fine-tuning |
| `8bit` (affine) | ~8,5 | ~53 % | Quasi identique | Oui | Quand la RAM le permet |
| `mxfp8` | ~8,25 | ~52 % | Quasi identique | Oui | Alternative flottante au 8bit |
| `6bit` | ~6,5 | ~40 % | Très proche | Oui | Bon compromis qualité et taille |
| `5bit` | ~5,5 | ~34 % | Très proche | Oui | Un cran au-dessus du 4bit |
| `4bit` (affine) | ~4,5 | ~28 % | Légère perte, souvent imperceptible en chat | Oui | **Le standard sur Mac** |
| `mxfp4` | ~4,25 | ~27 % | Comparable au 4bit | Oui | Modèles publiés en MXFP4 (gpt-oss…) |
| `nvfp4` | ~4,5 | ~28 % | Comparable au 4bit, souvent un peu meilleur | Oui | Alternative au 4bit affine |
| `3bit` | ~3,5 | ~22 % | Perte visible (raisonnement, code, faits précis) | Oui | Faire tenir un gros modèle |
| `2bit` | ~2,5 | ~16 % | Forte dégradation sans méthode avancée | Oui | Très gros modèles uniquement, avec DWQ/AWQ ou en mixte |
| `1,58bit` (ternaire −1/0/+1) | ~1,6-2 | ~10-12 % | Correcte **si le modèle a été entraîné ainsi** (BitNet b1.58) ; détruit un modèle classique | Non natif* | Modèles BitNet spécifiques |
| `1bit` (binaire ±1) | ~1-1,5 | ~6-9 % | Effondrement en quantification directe ; utilisable seulement après entraînement ou affinage dédié | Oui (affine 1 bit) | Expérimentation |
| `< 1 bit` (0,5-0,9) | 0,5-0,9 | ~3-6 % | Recherche : forte perte, même avec les meilleures méthodes | Non | Curiosité |

\* MLX peut stocker du ternaire sur 2 bits, mais sans format dédié. Les modèles BitNet tournent
plutôt avec `bitnet.cpp`.

« Affine » (le mode par défaut de MLX) : les poids sont découpés en groupes de 64
(`--q-group-size`), chaque groupe a son échelle et son biais en float16. D'où le ~0,5 bit de plus
par poids. Avec des groupes de 32 : +1 bit, meilleure qualité. Avec 128 : +0,25 bit, plus compact.

### Comment descend-on sous 2 bits ?

- **Ternaire et binaire entraînés** (BitNet b1.58, BitNet) : le modèle apprend dès l'entraînement
  avec des poids −1/0/+1. C'est la seule voie qui garde une qualité correcte à ~1,6 bit.
- **Binarisation post-entraînement** (BiLLM, ~1,1 bit) : binarise la plupart des poids et garde
  les poids critiques en plus haute précision.
- **Sous 1 bit, la quantification vectorielle et la structure** (codebooks à la AQLM, QuIP#, QTIP ;
  binarisation structurée à la STBLLM) : un code d'un dictionnaire représente un *groupe* de poids.
  Par exemple, 8 poids partagent un code de 4 bits, soit 0,5 bit par poids. Ces méthodes demandent
  des noyaux de calcul spécifiques, absents de MLX.

En pratique : **un gros modèle peu quantifié bat un petit modèle très quantifié jusqu'à ~3 bits**.
En dessous, la qualité chute vite, sauf pour les modèles conçus pour ces formats.

### Équivalences avec les noms GGUF (llama.cpp, Ollama, LM Studio)

Les noms GGUF ne sont pas lisibles par MLX, mais ils aident à comparer :

| GGUF | Bits effectifs | Proche de |
|---|---|---|
| `Q8_0` | ~8,5 | `8bit` |
| `Q6_K` | ~6,6 | `6bit` |
| `Q5_K_M` | ~5,7 | `5bit` |
| `Q4_K_M` | ~4,8 | `4bit` |
| `Q3_K_M` | ~3,9 | `3bit` |
| `IQ2_XS` / `Q2_K` | ~2,3-2,6 | `2bit` |
| `IQ1_S` / `IQ1_M` | ~1,6-1,8 | Sous 2 bits, forte perte |

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
4. En 2 bits ou moins, prends une version produite par une méthode avancée (DWQ, AWQ, mixte)
   ou un modèle entraîné pour ce format. Jamais un simple `-q --q-bits 2`.

Le poids en Go de chaque format, par taille de modèle, est dans [ram.md](ram.md).
