# Quels modèles sont installés sur ma machine ?

## Où sont-ils ?

`mlx_lm` et `mlx_vlm` téléchargent dans le **cache Hugging Face** :

```
~/.cache/huggingface/hub/models--<auteur>--<nom>/
├── blobs/            # les fichiers réels
├── refs/main         # le hash de la version courante
└── snapshots/<hash>/ # liens symboliques vers blobs/ : c'est ce dossier que MLX charge
```

L'emplacement change si `HF_HOME` ou `HF_HUB_CACHE` est défini. Le cache contient aussi des
modèles non MLX (embeddings `sentence-transformers`, modèles PyTorch bruts) téléchargés par
d'autres outils.

Les modèles convertis avec `mlx_lm.convert --mlx-path` sont là où tu les as mis (par exemple
`./models/`, ignoré par git). LM Studio range les siens dans `~/.lmstudio/models` : ce dossier
n'est pas couvert ici.

## Le script du projet (recommandé)

```bash
uv run python scripts/list_models.py          # modèles MLX seulement
uv run python scripts/list_models.py --all    # tout le cache
```

Pour chaque modèle : le type (`LLM`, `VLM` pour la vision, `/MoE`), la quantification, le
poids des fichiers de poids, la RAM estimée et un verdict. `oui` : tient confortablement.
`juste` : tient seul, autres apps fermées. `non` : dépasse le budget GPU. `incomplet` :
téléchargement interrompu.

Un modèle est considéré MLX si son `config.json` a une clé `quantization` ou si son nom
contient « mlx », quelle que soit la casse.

## Les commandes existantes

```bash
uv run mlx_lm.manage --scan                   # repos contenant "mlx", casse comprise
uv run mlx_lm.manage --scan --pattern MLX     # complète : attrape les ...-MLX-4bit
hf cache ls                                   # tout le cache, avec tailles et dates
du -sh ~/.cache/huggingface/hub/models--* | sort -h   # taille réelle sur disque
```

Piège : `mlx_lm.manage --scan` sans pattern n'affiche ici que 3 des 8 modèles MLX installés,
parce que le filtre « mlx » tient compte de la casse.

## Faire de la place

```bash
uv run mlx_lm.manage --delete --pattern LFM2-24B-A2B-MLX-8bit   # supprime les repos correspondants (avec confirmation)
hf cache rm model/lmstudio-community/LFM2-24B-A2B-MLX-8bit     # équivalent côté Hugging Face
hf cache prune                                                 # anciennes révisions et téléchargements incomplets
```

Un téléchargement incomplet se reprend en relançant n'importe quelle commande sur ce modèle,
par exemple `uv run mlx_lm.generate --model <repo> -p test`, ou avec `hf download <repo>`.

## Télécharger sans lancer le modèle

```bash
hf download mlx-community/gemma-4-e4b-it-4bit
```

Où chercher : <https://huggingface.co/mlx-community> (conversions MLX de référence). Sur la page
d'un modèle, l'onglet *Files* donne le poids total et donc, avec [ram.md](ram.md), s'il tiendra.
