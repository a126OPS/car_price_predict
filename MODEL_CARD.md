---
language:
- fr
license: mit
tags:
- tabular-regression
- price-prediction
- car
- joblib
- scikit-learn
metrics:
- rmse
- r2
---

# Car Price Predictor Model Card

## Description

Ce modele predit le prix de vente d'un vehicule d'occasion a partir de ses caracteristiques techniques et commerciales. Il est entraine sur des donnees de marche automobile francaises et vise a aider acheteurs et vendeurs a estimer un prix juste.

## Utilisation

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Chargement du modele
model_path = hf_hub_download(repo_id="a126OPS/Car_Predict", filename="model.joblib")
model = joblib.load(model_path)

# Exemple de prediction
features = np.array([[2019, 80000, 120, 1.6, 5]])  # annee, km, cv, cylindree, portes
predicted_price = model.predict(features)
print(f"Prix estime : {predicted_price[0]:.0f} EUR")
```

## Donnees d'entrainement

- **Source :** donnees de ventes de vehicules d'occasion
- **Variables d'entree :** annee, kilometrage, puissance (CV), type de carburant, marque, modele, boite de vitesses
- **Variable cible :** prix de vente en euros

## Limites

- Le modele est calibre sur le marche francais
- Les vehicules de luxe ou tres anciens peuvent etre moins bien estimes
- Les donnees d'entrainement ont une date limite ; les fluctuations recentes du marche ne sont pas refletees

## Auteur

Developpe par [a126OPS](https://huggingface.co/a126OPS)

Demo interactive : [car-price-predictor-demo](https://huggingface.co/spaces/a126OPS/car-price-predictor-demo)

## Licence

[MIT](https://opensource.org/licenses/MIT)
