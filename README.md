

## Pré‑requis
- Python ≥ 3.10
- pip + PyTorch ≥ 2 (CUDA optionnel)
- JupyterLab

## Installation
```bash
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
```

## Exécution du notebook
### Mode interactif
```bash
jupyter lab     # puis ouvrir PokerProject.ipynb
```
### Mode batch
```bash
jupyter nbconvert --execute --to html PokerProject.ipynb
```

## Scripts utiles
### Entraînement « self‑train » (blueprint)
```bash
python self_train.py   # entraîne la stratégie de référence
```
### Match contre Slumbot
```bash
python slumbotTest.py --username VOTRE_USER --password VOTRE_PASS
```

