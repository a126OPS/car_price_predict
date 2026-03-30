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

# 🚗 Car Price Predictor

## Description

Ce modèle prédit le **prix de vente d'un véhicule d'occasion** à partir de ses caractéristiques techniques et commerciales. Il est entraîné sur des données de marché automobile françaises et vise à aider acheteurs et vendeurs à estimer un prix juste.

## Utilisation

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Chargement du modèle
model_path = hf_hub_download(repo_id="a126OPS/Car_Predict", filename="model.joblib")
model = joblib.load(model_path)

# Exemple de prédiction
features = np.array([[2019, 80000, 120, 1.6, 5]])  # année, km, cv, cylindrée, portes
predicted_price = model.predict(features)
print(f"Prix estimé : {predicted_price[0]:.0f} €")
```

## Données d'entraînement

- **Source :** Données de ventes de véhicules d'occasion
- **Variables d'entrée :** année, kilométrage, puissance (CV), type de carburant, marque, modèle, boîte de vitesses
- **Variable cible :** prix de vente en euros

## Limites

- Le modèle est calibré sur le marché français
- Les véhicules de luxe ou très anciens peuvent être moins bien estimés
- Les données d'entraînement ont une date limite — les fluctuations récentes du marché ne sont pas reflétées

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Démo interactive : [car-price-predictor-demo](https://huggingface.co/spaces/a126OPS/car-price-predictor-demo)

## Licence

[MIT](https://opensource.org/licenses/MIT)
