# RAM : combien en faut-il pour un modèle ?

## Mémoire unifiée et limite du GPU

Sur Apple Silicon, CPU et GPU partagent la même RAM : pas de VRAM séparée. MLX charge les
poids une seule fois, directement là où le GPU les lit.

Mais macOS ne laisse pas tout au GPU. Par défaut, la limite (*recommended working set*)
tourne autour de **70 à 75 % de la RAM** :

| RAM du Mac | Budget GPU approximatif |
|---|---|
| 16 Go | ~11 Go |
| 24 Go | **19,1 Go** (mesuré sur ton M3) |
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

La quantification MLX (mode `affine`) ajoute une échelle et un biais en float16 par groupe de
64 poids, soit ~0,5 bit de plus par poids : un « 4bit » coûte en réalité ~4,5 bits. Avec
`--q-group-size 32`, c'est +1 bit ; avec 128, +0,25 bit.

Poids en Go selon le format (bits effectifs, surcoût des échelles compris) :

| Format | Bits eff. | 1B | 3B | 8B | 14B | 24B | 32B | 70B | 120B | 235B | 405B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 32 | 4 | 12 | 32 | 56 | 96 | 128 | 280 | 480 | 940 | 1620 |
| bf16 / fp16 | 16 | 2 | 6 | 16 | 28 | 48 | 64 | 140 | 240 | 470 | 810 |
| 8bit | 8,5 | 1,1 | 3,2 | 8,5 | 15 | 26 | 34 | 74 | 128 | 250 | 430 |
| mxfp8 | 8,25 | 1 | 3,1 | 8,2 | 14 | 25 | 33 | 72 | 124 | 242 | 418 |
| 6bit | 6,5 | 0,8 | 2,4 | 6,5 | 11 | 20 | 26 | 57 | 98 | 191 | 329 |
| 5bit | 5,5 | 0,7 | 2,1 | 5,5 | 9,6 | 16 | 22 | 48 | 82 | 162 | 278 |
| 4bit | 4,5 | 0,6 | 1,7 | 4,5 | 7,9 | 14 | 18 | 39 | 68 | 132 | 228 |
| nvfp4 | 4,5 | 0,6 | 1,7 | 4,5 | 7,9 | 14 | 18 | 39 | 68 | 132 | 228 |
| mxfp4 | 4,25 | 0,5 | 1,6 | 4,2 | 7,4 | 13 | 17 | 37 | 64 | 125 | 215 |
| 3bit | 3,5 | 0,4 | 1,3 | 3,5 | 6,1 | 10 | 14 | 31 | 52 | 103 | 177 |
| 2bit | 2,5 | 0,3 | 0,9 | 2,5 | 4,4 | 7,5 | 10 | 22 | 38 | 73 | 127 |
| 1,58bit (ternaire) | ~2 | 0,2 | 0,8 | 2 | 3,5 | 6 | 8 | 18 | 30 | 59 | 101 |
| 1bit | ~1,5 | 0,2 | 0,6 | 1,5 | 2,6 | 4,5 | 6 | 13 | 22 | 44 | 76 |
| < 1 bit (codebooks) | 0,5-1 | — | — | 0,5-1 | 0,9-1,8 | 1,5-3 | 2-4 | 4-9 | 8-15 | 15-29 | 25-51 |

Lecture : 4bit ≈ **0,56 Go par milliard de paramètres**, 8bit ≈ 1,06 Go, bf16 = 2 Go.

- MLX quantifie nativement en 1, 2, 3, 4, 5, 6 et 8 bits, ainsi qu'en mxfp4, nvfp4 et mxfp8.
- Le ternaire (1,58 bit, BitNet) se stocke en pratique sur ~2 bits. Il n'est utile que pour les
  modèles **entraînés** ainsi : quantifier un modèle classique en ternaire le détruit.
- En 1 bit, le surcoût des échelles devient énorme (+0,5 bit pour 1 bit utile) : la qualité
  s'effondre sauf pour un modèle entraîné ou affiné pour ce format.
- Sous 1 bit, il faut une quantification vectorielle (plusieurs poids partagent un code d'un
  dictionnaire). Ces formats existent en recherche mais ne sont pas pris en charge par MLX.
  Le détail est dans [quantification.md](quantification.md).
- Un modèle « Xbit » publié est souvent **mixte** (couches sensibles et embeddings en plus de
  bits) : il pèse un peu plus que le tableau. La taille réelle des fichiers fait foi.

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
`head_dim`). Exemple pour un 8B classique (32 couches, 8 têtes KV, dimension 128) :
128 Ko par token en float16.

Par défaut, le KV cache est stocké dans le type du modèle (bf16 ou fp16). `--kv-bits N`
le quantifie, avec le même surcoût d'échelles que les poids (+0,5 bit par groupe de 64).

| Format du KV cache | Option | Par token | 4K tokens | 8K | 32K | 128K | 1M |
|---|---|---|---|---|---|---|---|
| fp32 | — (rare) | 256 Ko | 1,1 Go | 2,1 Go | 8,6 Go | 34 Go | 275 Go |
| bf16 / fp16 | défaut | 128 Ko | 0,54 Go | 1,1 Go | 4,3 Go | 17 Go | 137 Go |
| 8bit | `--kv-bits 8` | 68 Ko | 0,29 Go | 0,57 Go | 2,3 Go | 9,1 Go | 73 Go |
| 6bit | `--kv-bits 6` | 52 Ko | 0,22 Go | 0,44 Go | 1,7 Go | 7 Go | 56 Go |
| 4bit | `--kv-bits 4` | 36 Ko | 0,15 Go | 0,3 Go | 1,2 Go | 4,8 Go | 39 Go |
| 3bit | `--kv-bits 3` | 28 Ko | 0,12 Go | 0,23 Go | 0,94 Go | 3,8 Go | 30 Go |
| 2bit | `--kv-bits 2` | 20 Ko | 0,08 Go | 0,17 Go | 0,67 Go | 2,7 Go | 21 Go |

- **8bit** : perte de qualité négligeable. C'est le premier levier à activer sur un long contexte.
- **4bit** : bon compromis, avec une légère dégradation sur la mémoire de détails lointains.
- **3 et 2 bits** : dégradation nette, à réserver aux contextes extrêmes.
- `mlx_vlm` propose aussi TurboQuant (`--kv-quant-scheme turboquant`), qui accepte des bits
  fractionnaires comme `--kv-bits 3.5`.
- `--quantized-kv-start N` garde les N premiers tokens en pleine précision.
- **Limite de `mlx_lm`** : `--kv-bits` échoue sur les modèles à fenêtre glissante (Gemma 3 et 4…)
  avec `RotatingKVCache Quantization NYI`. Pour eux, seul `--max-kv-size` borne la mémoire.
- Ces chiffres varient beaucoup d'un modèle à l'autre. Les architectures récentes réduisent
  fortement le KV cache : moins de têtes KV (GQA), attention à fenêtre glissante sur une partie
  des couches, couches linéaires ou hybrides. Recalcule toujours avec le `config.json` du modèle.

Un modèle qui « tient » à vide peut donc saturer sur un long document. Les leviers pour l'éviter
sont dans [contexte-performance.md](contexte-performance.md).

### 3. Le pic du prefill

Pendant le traitement d'un long prompt, les activations intermédiaires ajoutent un pic
temporaire. `--prefill-step-size 512` le réduit, au prix d'un prefill plus lent.

## Ce qui tient sur ton Mac (M3, 24 Go, budget GPU 19,1 Go)

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
