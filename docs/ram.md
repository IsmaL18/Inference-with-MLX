# RAM : combien en faut-il pour un modèle ?

## Mémoire unifiée et limite du GPU

Sur Apple Silicon, CPU et GPU partagent la même RAM : pas de VRAM séparée. MLX charge les
poids une seule fois, directement là où le GPU les lit.

Mais macOS ne laisse pas tout au GPU. Par défaut, la limite (*recommended working set*)
tourne autour de **70 à 75 % de la RAM** :

| RAM du Mac | Budget GPU approximatif |
|---|---|
| 16 Go | ~11 Go |
| 24 Go | **19,1 Go** (mesuré sur ton M4) |
| 32 Go | ~22-24 Go |
| 64 Go | ~48 Go |
| 128 Go | ~96 Go |

Ton budget exact :

```bash
uv run python -c "import mlx.core as mx; print(mx.device_info()['max_recommended_working_set_size']/1e9, 'Go')"
```

Au-delà, macOS swappe sur le SSD : la génération s'effondre, voire la machine gèle.

### Relever la limite (à tes risques)

```bash
sysctl iogpu.wired_limit_mb                 # 0 = valeur par défaut de macOS
sudo sysctl iogpu.wired_limit_mb=20480      # 20 Go pour le GPU sur un Mac de 24 Go
```

Le réglage saute au redémarrage. Laisse **au moins 4 Go** au système et aux apps, et ferme
navigateur et IDE avant de charger un gros modèle.

## La règle de calcul

**RAM nécessaire ≈ poids du modèle + KV cache + ~0,5-1 Go de marge**

### 1. Les poids

Le poids en Go est à peu près égal à la RAM qu'occupent les poids. Pour l'estimer à partir
du nombre de paramètres :

**Poids (Go) ≈ paramètres (milliards) × bits par poids ÷ 8**

La quantification MLX ajoute une échelle et un biais par groupe de 64 poids, soit ~0,5 bit
de plus par poids : un « 4bit » coûte en réalité ~4,5 bits.

| Modèle | float16 | 8bit | 4bit | 3bit |
|---|---|---|---|---|
| 3B | 6 Go | 3,2 Go | 1,7 Go | 1,3 Go |
| 8B | 16 Go | 8,5 Go | 4,5 Go | 3,5 Go |
| 14B | 28 Go | 15 Go | 8 Go | 6 Go |
| 24B | 48 Go | 25,5 Go | 13,5 Go | 10,5 Go |
| 32B | 64 Go | 34 Go | 18 Go | 14 Go |
| 35B | 70 Go | 37 Go | 20 Go | 15,5 Go |
| 70B | 140 Go | 75 Go | 40 Go | 31 Go |

**Mixture of Experts (MoE)** : un `35B-A3B` a 35 milliards de paramètres dont 3 milliards
*actifs* par token. **Tous** les experts doivent tenir en RAM : il faut donc compter 35B pour la
mémoire. Les 3B actifs déterminent seulement la vitesse, proche de celle d'un modèle de 3B.

**Poids sur disque et RAM** : un modèle multimodal stocke aussi ses encodeurs vision et audio.
Avec `mlx_lm`, en texte seul, ils ne sont pas chargés. Mesuré : `gemma-4-e4b-it-4bit` pèse
5,1 Go sur disque pour un pic de 4,3 Go.

### 2. Le KV cache, qui grandit avec le contexte

À chaque token du contexte (prompt + réponse), le modèle garde ses clés et valeurs d'attention :

**KV cache (octets) = 2 × couches × têtes KV × dimension de tête × octets par valeur × tokens**

Ces valeurs sont dans le `config.json` du modèle (`num_hidden_layers`, `num_key_value_heads`,
`head_dim`). Exemple pour un 8B classique (32 couches, 8 têtes KV, dimension 128, float16) :
128 Ko par token.

| Contexte | KV cache (float16) | Avec `--kv-bits 8` | Avec `--kv-bits 4` |
|---|---|---|---|
| 4 000 tokens | 0,5 Go | 0,25 Go | 0,15 Go |
| 32 000 tokens | 4 Go | 2 Go | 1,1 Go |
| 128 000 tokens | 16 Go | 8 Go | 4,5 Go |

Un modèle qui « tient » à vide peut donc saturer sur un long document. Les leviers pour l'éviter
sont dans [contexte-performance.md](contexte-performance.md).

### 3. Le pic du prefill

Pendant le traitement d'un long prompt, les activations intermédiaires ajoutent un pic
temporaire. `--prefill-step-size 512` le réduit, au prix d'un prefill plus lent.

## Ce qui tient sur ton Mac (M4, 24 Go, budget GPU 19,1 Go)

Hypothèse : contexte modéré (quelques milliers de tokens), autres apps légères.

| Poids du modèle | Verdict | Exemples |
|---|---|---|
| ≤ 10 Go | Confortable, même avec un long contexte | 8B en 8bit, 14B en 4bit, gemma-4-e4b |
| 10-15 Go | OK si tu fermes les apps gourmandes | 24B en 4bit (LFM2-24B-A2B-4bit : 13,4 Go) |
| 15-18 Go | Limite : petit contexte, apps fermées, ou limite GPU relevée | 32B en 4bit, 35B-A3B en 3bit (17,4 Go) |
| > 18 Go | Ne tient pas (swap) | 35B en 4bit (21 Go), 24B en 8bit (25,3 Go) |

`uv run python scripts/list_models.py` calcule ce verdict pour chaque modèle installé
(`RAM est. = poids × 1,15 + 0,5 Go`). C'est une estimation volontairement prudente, pour un
contexte modéré.

## Mesurer au lieu d'estimer

- `mlx_lm.generate` affiche `Peak memory` en fin de génération.
- `examples/02_stream.py` affiche le pic mémoire.
- Moniteur d'activité, onglet Mémoire : la « pression mémoire » doit rester verte. Si elle
  passe en jaune ou rouge et que le swap grossit, le modèle est trop gros pour la situation.
