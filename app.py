import os
import sys
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import requests
from sklearn.base import BaseEstimator, TransformerMixin


APP_TITLE = "Estimateur de Prix Voiture"
APP_DESCRIPTION = (
    "API REST et interface Gradio pour estimer le prix d'un vehicule d'occasion."
)
MODEL_PATH_XGB = "best_pipeline_xgb.joblib"
MODEL_PATH_LR = "best_pipeline_lr.joblib"
PRICE_MAE_EUR = 4500
MARKET_MEDIAN_EUR = 23000
HF_MODEL_SOURCES = [
    ("a126OPS/Car_Predict", "model"),
    ("a126OPS/car-price-predictor-demo", "space"),
]

BRAND_OPTIONS = [
    "Toyota",
    "BMW",
    "Mercedes",
    "Renault",
    "Peugeot",
    "Audi",
    "Ford",
    "Volkswagen",
    "Honda",
    "Nissan",
]
FUEL_OPTIONS = ["Essence", "Diesel", "Hybride", "Electrique"]
TRANSMISSION_OPTIONS = ["Manuelle", "Automatique"]
STATE_OPTIONS = ["Mauvais", "Passable", "Bon", "Tres bon", "Neuf"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer stored in the serialized pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_vehicule"] = 2024 - X["annee"]
        X["km_par_an"] = X["kilometrage"] / (X["age_vehicule"] + 1)
        X["puissance_par_litre"] = X["puissance_cv"] / X["consommation_L100km"]
        X["est_recente"] = (X["age_vehicule"] <= 5).astype(int)
        return X.drop(columns=["annee"])


# Required so joblib can resolve __main__.FeatureEngineer while unpickling.
sys.modules.setdefault("__main__", sys.modules[__name__]).FeatureEngineer = FeatureEngineer


class CarPredictionRequest(BaseModel):
    marque: str
    annee: int = Field(..., ge=2005, le=2024)
    kilometrage: float = Field(..., ge=0)
    puissance_cv: float = Field(..., ge=70, le=400)
    nb_portes: int = Field(..., ge=2, le=5)
    carburant: str
    transmission: str
    etat: str
    nb_proprietaires: int = Field(..., ge=1, le=5)
    consommation_L100km: float = Field(..., ge=0.0, le=15.0)


class PriceRangeResponse(BaseModel):
    minimum: float
    estimate: float
    maximum: float
    mae_eur: float


class MarketVerdictResponse(BaseModel):
    label: str
    advice: str


class VehicleSummaryResponse(BaseModel):
    marque: str
    annee: int
    kilometrage: float
    puissance_cv: float
    nb_portes: int
    carburant: str
    transmission: str
    etat: str
    nb_proprietaires: int
    consommation_L100km: float
    km_par_an: float


class CarPredictionResponse(BaseModel):
    model_name: str
    currency: str
    predicted_price: float
    price_range: PriceRangeResponse
    market_verdict: MarketVerdictResponse
    vehicle: VehicleSummaryResponse
    summary_markdown: str


def load_pipeline() -> tuple[Any, str]:
    if os.path.exists(MODEL_PATH_XGB):
        return joblib.load(MODEL_PATH_XGB), "XGBoost"
    if os.path.exists(MODEL_PATH_LR):
        return joblib.load(MODEL_PATH_LR), "Regression lineaire"

    for repo_id, repo_type in HF_MODEL_SOURCES:
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=MODEL_PATH_XGB,
            )
            return joblib.load(cached_path), "XGBoost"
        except Exception:
            pass

        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=MODEL_PATH_LR,
            )
            return joblib.load(cached_path), "Regression lineaire"
        except Exception:
            pass

    raise FileNotFoundError(
        "Aucun modele trouve localement ou sur Hugging Face. "
        "Ajoutez best_pipeline_xgb.joblib / best_pipeline_lr.joblib ou verifiez les repos distants."
    )


pipeline, MODEL_NAME = load_pipeline()
print(f"[OK] Modele charge : {MODEL_NAME}")


def parse_allowed_origins() -> list[str]:
    raw_value = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if not raw_value:
        return ["*"]
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or ["*"]


def validate_choice(field_name: str, value: str, allowed_values: list[str]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} invalide: {value}. Valeurs autorisees: {allowed}.")


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "marque": str(payload["marque"]),
        "annee": int(payload["annee"]),
        "kilometrage": float(payload["kilometrage"]),
        "puissance_cv": float(payload["puissance_cv"]),
        "nb_portes": int(payload["nb_portes"]),
        "carburant": str(payload["carburant"]),
        "transmission": str(payload["transmission"]),
        "etat": str(payload["etat"]),
        "nb_proprietaires": int(payload["nb_proprietaires"]),
        "consommation_L100km": float(payload["consommation_L100km"]),
    }

    validate_choice("marque", normalized["marque"], BRAND_OPTIONS)
    validate_choice("carburant", normalized["carburant"], FUEL_OPTIONS)
    validate_choice("transmission", normalized["transmission"], TRANSMISSION_OPTIONS)
    validate_choice("etat", normalized["etat"], STATE_OPTIONS)

    return normalized


def build_vehicle_frame(car: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([car])


def get_market_verdict(predicted_price: float) -> tuple[str, str]:
    if predicted_price < MARKET_MEDIAN_EUR * 0.85:
        return "BONNE AFFAIRE", "Ce prix est en dessous de la mediane du marche."
    if predicted_price < MARKET_MEDIAN_EUR * 1.15:
        return "PRIX CORRECT", "Ce prix est dans la fourchette normale du marche."
    return "PRIX ELEVE", "Ce prix est au-dessus de la mediane du marche."


def build_summary_markdown(result: dict[str, Any]) -> str:
    vehicle = result["vehicle"]
    verdict = result["market_verdict"]
    price_range = result["price_range"]

    return f"""
## Prix estime : **{result["predicted_price"]:,.0f} EUR**

---

### Fourchette de prix
| Pessimiste | Estimation | Optimiste |
|:----------:|:----------:|:---------:|
| {price_range["minimum"]:,.0f} EUR | **{price_range["estimate"]:,.0f} EUR** | {price_range["maximum"]:,.0f} EUR |

---

### Verdict marche
**{verdict["label"]}**
> {verdict["advice"]}

---

### Recapitulatif voiture
| Caracteristique | Valeur |
|:----------------|:-------|
| Marque / Annee  | {vehicle["marque"]} {vehicle["annee"]} |
| Kilometrage     | {vehicle["kilometrage"]:,.0f} km |
| Km/an moyen     | {vehicle["km_par_an"]:,.0f} km/an |
| Puissance       | {vehicle["puissance_cv"]:,.0f} cv |
| Carburant       | {vehicle["carburant"]} ({vehicle["transmission"]}) |
| Etat            | {vehicle["etat"]} |
| Proprietaires   | {vehicle["nb_proprietaires"]} |
| Consommation    | {vehicle["consommation_L100km"]:.1f} L/100km |

---
*Modele : {result["model_name"]} - Marge d'erreur : +/-{price_range["mae_eur"]:,.0f} EUR*
"""


def predict_price_details(payload: dict[str, Any]) -> dict[str, Any]:
    car = normalize_payload(payload)
    car_frame = build_vehicle_frame(car)

    predicted_price = max(500.0, float(pipeline.predict(car_frame)[0]))
    price_min = max(0.0, predicted_price - PRICE_MAE_EUR)
    price_max = predicted_price + PRICE_MAE_EUR
    verdict_label, verdict_advice = get_market_verdict(predicted_price)

    current_year = datetime.now().year
    vehicle_age = max(current_year - car["annee"], 1)
    km_per_year = car["kilometrage"] / vehicle_age

    result = {
        "model_name": MODEL_NAME,
        "currency": "EUR",
        "predicted_price": round(predicted_price, 2),
        "price_range": {
            "minimum": round(price_min, 2),
            "estimate": round(predicted_price, 2),
            "maximum": round(price_max, 2),
            "mae_eur": float(PRICE_MAE_EUR),
        },
        "market_verdict": {
            "label": verdict_label,
            "advice": verdict_advice,
        },
        "vehicle": {
            **car,
            "km_par_an": round(km_per_year, 2),
        },
    }
    result["summary_markdown"] = build_summary_markdown(result)
    return result


def predire_prix(
    marque,
    annee,
    kilometrage,
    puissance_cv,
    nb_portes,
    carburant,
    transmission,
    etat,
    nb_proprietaires,
    consommation,
):
    result = predict_price_details(
        {
            "marque": marque,
            "annee": annee,
            "kilometrage": kilometrage,
            "puissance_cv": puissance_cv,
            "nb_portes": nb_portes,
            "carburant": carburant,
            "transmission": transmission,
            "etat": etat,
            "nb_proprietaires": nb_proprietaires,
            "consommation_L100km": consommation,
        }
    )
    return result["predicted_price"], result["summary_markdown"]


def predire_prix_carburant(departement, jour):
    """Proxy helper kept for the secondary tab."""
    url = "https://huggingface.co/spaces/a126OPS/carburant_predict"
    payload = {"departement": departement, "jour": jour}

    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        return "Erreur", f"Service carburant indisponible: {exc}"

    return result.get("prix", "Erreur"), result.get("details", "")


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(),
        css="""
            :root {
                --bg: #0b1021;
                --panel: #111827;
                --text: #e5e7eb;
                --muted: #9ca3af;
                --accent: #facc15;
            }
            body, .gradio-container { background: var(--bg); color: var(--text); }
            .gradio-container { max-width: 900px !important; }
            footer { display: none !important; }
            #result-box {
                background: var(--panel);
                color: var(--text);
                border: 1px solid #1f2937;
                border-radius: 14px;
                padding: 1.25rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            }
            #result-box h2 strong { color: var(--accent); }
            #result-box table { color: var(--text); }
            #result-box em, #result-box small { color: var(--muted); }
            button { background: var(--accent) !important; color: #111827 !important; }
            input, select, textarea { background: #0f172a !important; color: var(--text) !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # Estimateur de Prix Voiture
            **Modele ML entraine sur des donnees reelles du marche francais**

            Remplissez le formulaire ci-dessous pour obtenir une estimation instantanee du prix d'un vehicule.

            API JSON disponible sur `/api/predict` et documentation automatique sur `/docs`.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Informations generales")
                marque = gr.Dropdown(choices=BRAND_OPTIONS, label="Marque", value="Renault")
                annee = gr.Slider(minimum=2005, maximum=2024, value=2018, step=1, label="Annee")
                kilometrage = gr.Number(value=80000, label="Kilometrage (km)", minimum=0)
                nb_portes = gr.Radio(choices=[2, 3, 4, 5], value=4, label="Nombre de portes")
                nb_proprietaires = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Nombre de proprietaires",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Caracteristiques techniques")
                puissance_cv = gr.Slider(minimum=70, maximum=400, value=120, step=5, label="Puissance (cv)")
                carburant = gr.Dropdown(choices=FUEL_OPTIONS, label="Carburant", value="Essence")
                transmission = gr.Radio(
                    choices=TRANSMISSION_OPTIONS,
                    value="Manuelle",
                    label="Transmission",
                )
                etat = gr.Dropdown(choices=STATE_OPTIONS, value="Bon", label="Etat du vehicule")
                consommation = gr.Slider(
                    minimum=0.0,
                    maximum=15.0,
                    value=6.5,
                    step=0.1,
                    label="Consommation (L/100km)",
                )

        gr.Markdown("---")

        btn = gr.Button("Estimer le prix", variant="primary", size="lg")
        prix_brut = gr.Number(label="Prix estime (EUR)", precision=0, interactive=False)
        output = gr.Markdown(
            value="*Remplissez le formulaire et cliquez sur **Estimer le prix**...*",
            elem_id="result-box",
        )

        btn.click(
            fn=predire_prix,
            inputs=[
                marque,
                annee,
                kilometrage,
                puissance_cv,
                nb_portes,
                carburant,
                transmission,
                etat,
                nb_proprietaires,
                consommation,
            ],
            outputs=[prix_brut, output],
            api_name="predict_price",
        )

        with gr.Tab("Prediction Prix Carburant"):
            gr.Markdown(
                """
                # Prediction des Prix Carburant a J+7
                **Modele connecte au flux officiel des prix carburant.**
                """
            )

            with gr.Row():
                departement = gr.Textbox(label="Departement", placeholder="Exemple : 75")
                jour = gr.Slider(minimum=1, maximum=7, value=1, step=1, label="Jour (J+)")

            btn_carburant = gr.Button("Predire", variant="primary")
            prix_carburant = gr.Textbox(label="Prix estime (EUR)", interactive=False)
            details_carburant = gr.Markdown("*Cliquez sur Predire pour voir les details...*")

            btn_carburant.click(
                fn=predire_prix_carburant,
                inputs=[departement, jour],
                outputs=[prix_carburant, details_carburant],
                api_name="predict_fuel_price",
            )

    return demo


demo = build_gradio_app()

api = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version="1.0.0")
allowed_origins = parse_allowed_origins()

api.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/api/health")
def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "api_version": "1.0.0",
    }


@api.get("/api/options")
def api_options() -> dict[str, Any]:
    return {
        "brands": BRAND_OPTIONS,
        "fuel_types": FUEL_OPTIONS,
        "transmissions": TRANSMISSION_OPTIONS,
        "vehicle_states": STATE_OPTIONS,
        "model_name": MODEL_NAME,
        "docs_url": "/docs",
        "predict_url": "/api/predict",
    }


@api.post("/api/predict", response_model=CarPredictionResponse)
def api_predict(payload: CarPredictionRequest) -> CarPredictionResponse:
    try:
        result = predict_price_details(payload.model_dump())
        return CarPredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)
