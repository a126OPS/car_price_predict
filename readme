# Car Price Predictor

Car Price Predictor is a Gradio app that estimates the selling price of a used car from its main technical and commercial characteristics. The app returns an estimated price, a price range, and a quick market verdict.

## Main Features

- Estimate the price of a used vehicle with a trained machine learning pipeline
- Display a pessimistic, central, and optimistic price range
- Show a readable vehicle recap with mileage per year
- Provide a Gradio interface ready for local use or deployment
- Include a secondary tab connected to a fuel-price prediction service

## Model Inputs

- `marque`
- `annee`
- `kilometrage`
- `puissance_cv`
- `nb_portes`
- `carburant`
- `transmission`
- `etat`
- `nb_proprietaires`
- `consommation_L100km`

## Tech Stack

- Python
- Gradio
- pandas
- scikit-learn
- XGBoost
- joblib

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Add one of these model files at the project root before starting the app:

- `best_pipeline_xgb.joblib`
- `best_pipeline_lr.joblib`

Launch the interface:

```bash
python app.py
```

## Demo

https://huggingface.co/spaces/a126OPS/car-price-predictor-demo

## Author

Atillio Houngue

Portfolio: https://atillio-houngue.github.io
