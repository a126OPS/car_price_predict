# Car Price Predictor

Car Price Predictor is a Gradio app that estimates the selling price of a used car from its main technical and commercial characteristics. The app returns an estimated price, a price range, and a quick market verdict.

## Project Documents

- Main project page: this `README.md`
- Model card in Markdown: [MODEL_CARD.md](./MODEL_CARD.md)
- Original root file: [readme](./readme)

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

If the model files are missing locally, the app automatically tries to download them from the Hugging Face repositories:

- `a126OPS/Car_Predict`
- `a126OPS/car-price-predictor-demo`

Launch the interface:

```bash
python app.py
```

This starts both:

- the Gradio interface at `/`
- the API documentation at `/docs`

## API Endpoints

- `GET /api/health`
- `GET /api/options`
- `POST /api/predict`

Example request body:

```json
{
  "marque": "Renault",
  "annee": 2018,
  "kilometrage": 80000,
  "puissance_cv": 120,
  "nb_portes": 4,
  "carburant": "Essence",
  "transmission": "Manuelle",
  "etat": "Bon",
  "nb_proprietaires": 1,
  "consommation_L100km": 6.5
}
```

## Portfolio Integration

You can call the deployed API directly from your portfolio with `fetch()`:

```js
const response = await fetch("https://your-space-url.hf.space/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    marque: "Renault",
    annee: 2018,
    kilometrage: 80000,
    puissance_cv: 120,
    nb_portes: 4,
    carburant: "Essence",
    transmission: "Manuelle",
    etat: "Bon",
    nb_proprietaires: 1,
    consommation_L100km: 6.5
  })
});

const data = await response.json();
console.log(data.predicted_price);
console.log(data.summary_markdown);
```

If you need to restrict CORS to your portfolio domain, set the `ALLOWED_ORIGINS` environment variable with a comma-separated list of allowed origins.

## Demo

https://huggingface.co/spaces/a126OPS/car-price-predictor-demo

## Author

Atillio Houngue

Portfolio: https://atillio-houngue.github.io
