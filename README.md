---
title: Prédiction Prix Carburant sur 7 jours
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Modèle de prédiction J+7 des prix carburant à la pompe par département français.
---

# Prédiction Prix Carburant sur 7 jours

Modèle de prédiction J+7 des prix carburant à la pompe par département français. 14 millions d'observations réelles (data.economie.gouv.fr), feature engineering séries temporelles, MAE de 2 centimes. Interface temps réel connectée au flux officiel.

## Modèle

- Algorithme : XGBoost Regressor (pipeline Scikit-learn)
- Métriques : MAE ≈ 2 centimes
- Données : 14 millions d'observations réelles (data.economie.gouv.fr)

## Utilisation

Remplissez le formulaire avec les caractéristiques demandées et cliquez sur **Prédire** pour obtenir une estimation des prix carburant à J+7.

## Auteur

**Atillio HOUNGUE** — Data Scientist & ML Engineer  
Email : atilliohoungue@gmail.com  
Portfolio : https://atillio-houngue.github.io
