import os
import sys
from datetime import datetime

import gradio as gr
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import requests


# Custom transformer used inside the saved pipelines
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_vehicule"] = 2024 - X["annee"]
        X["km_par_an"] = X["kilometrage"] / (X["age_vehicule"] + 1)
        X["puissance_par_litre"] = X["puissance_cv"] / X["consommation_L100km"]
        X["est_recente"] = (X["age_vehicule"] <= 5).astype(int)
        return X.drop(columns=["annee"])


# Make the class discoverable during unpickling (__main__.FeatureEngineer)
sys.modules.setdefault("__main__", sys.modules[__name__]).FeatureEngineer = FeatureEngineer


# Model loading (prefers XGBoost, falls back to linear regression)
MODEL_PATH_XGB = "best_pipeline_xgb.joblib"
MODEL_PATH_LR = "best_pipeline_lr.joblib"

if os.path.exists(MODEL_PATH_XGB):
    pipeline = joblib.load(MODEL_PATH_XGB)
    MODEL_NAME = "XGBoost"
elif os.path.exists(MODEL_PATH_LR):
    pipeline = joblib.load(MODEL_PATH_LR)
    MODEL_NAME = "Régression linéaire"
else:
    raise FileNotFoundError("Aucun modèle trouvé. Ajoutez best_pipeline_xgb.joblib ou best_pipeline_lr.joblib.")

print(f"[OK] Modèle chargé : {MODEL_NAME}")


# Ajouter une fonction pour interagir avec le modèle Hugging Face
def predire_prix_carburant(departement, jour):
    """Interagit avec le modèle Hugging Face pour prédire les prix carburant."""
    url = "https://huggingface.co/spaces/a126OPS/carburant_predict"
    payload = {"departement": departement, "jour": jour}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result.get("prix", "Erreur : réponse invalide"), result.get("details", "")
    else:
        return "Erreur", f"Code HTTP : {response.status_code}"


def predire_prix(
    marque, annee, kilometrage, puissance_cv,
    nb_portes, carburant, transmission,
    etat, nb_proprietaires, consommation
):
    """Retourne le prix estimé et un récapitulatif formaté."""
    voiture = pd.DataFrame([{
        "marque": marque,
        "annee": int(annee),
        "kilometrage": float(kilometrage),
        "puissance_cv": float(puissance_cv),
        "nb_portes": int(nb_portes),
        "carburant": carburant,
        "transmission": transmission,
        "etat": etat,
        "nb_proprietaires": int(nb_proprietaires),
        "consommation_L100km": float(consommation),
    }])

    prix = float(pipeline.predict(voiture)[0])
    prix = max(500.0, prix)  # garde-fou

    mae = 4500
    prix_min = max(0.0, prix - mae)
    prix_max = prix + mae

    med = 23000
    if prix < med * 0.85:
        scoring = "BONNE AFFAIRE"
        conseil = "Ce prix est en dessous de la médiane du marché."
    elif prix < med * 1.15:
        scoring = "PRIX CORRECT"
        conseil = "Ce prix est dans la fourchette normale du marché."
    else:
        scoring = "PRIX ÉLEVÉ"
        conseil = "Ce prix est au-dessus de la médiane du marché."

    age = datetime.now().year - int(annee)
    km_par_an = float(kilometrage) / max(age, 1)

    recap = f"""
## Prix estimé : **{prix:,.0f} €**

---

### Fourchette de prix
| Pessimiste | **Estimation** | Optimiste |
|:----------:|:--------------:|:---------:|
| {prix_min:,.0f} € | **{prix:,.0f} €** | {prix_max:,.0f} € |

---

### Verdict marché
**{scoring}**
> {conseil}

---

### Récapitulatif voiture
| Caractéristique | Valeur |
|:----------------|:-------|
| Marque / Année  | {marque} {int(annee)} |
| Kilométrage     | {float(kilometrage):,.0f} km |
| Km/an moyen     | {km_par_an:,.0f} km/an |
| Puissance       | {int(puissance_cv)} cv |
| Carburant       | {carburant} ({transmission}) |
| État            | {etat} |
| Propriétaires   | {int(nb_proprietaires)} |
| Consommation    | {float(consommation)} L/100km |

---
*Modèle : {MODEL_NAME} — Marge d'erreur : ±{mae:,} €*
"""
    return prix, recap


with gr.Blocks(
    title="Estimateur de Prix Voiture — Atillio HOUNGUE",
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
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        #result-box h2 strong { color: var(--accent); }
        #result-box table { color: var(--text); }
        #result-box em, #result-box small { color: var(--muted); }
        button { background: var(--accent) !important; color: #111827 !important; }
        input, select, textarea { background: #0f172a !important; color: var(--text) !important; }
    """
) as demo:

    gr.Markdown("""
    # Estimateur de Prix Voiture
    **Modèle ML entraîné sur des données réelles du marché français**
    Remplissez le formulaire ci-dessous pour obtenir une estimation instantanée du prix d'un véhicule.
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Informations générales")
            marque = gr.Dropdown(
                choices=["Toyota", "BMW", "Mercedes", "Renault", "Peugeot",
                         "Audi", "Ford", "Volkswagen", "Honda", "Nissan"],
                label="Marque", value="Renault"
            )
            annee = gr.Slider(minimum=2005, maximum=2024, value=2018, step=1, label="Année")
            kilometrage = gr.Number(value=80000, label="Kilométrage (km)", minimum=0)
            nb_portes = gr.Radio(choices=[2, 3, 4, 5], value=4, label="Nombre de portes")
            nb_proprietaires = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Nombre de propriétaires")

        with gr.Column(scale=1):
            gr.Markdown("### Caractéristiques techniques")
            puissance_cv = gr.Slider(minimum=70, maximum=400, value=120, step=5, label="Puissance (cv)")
            carburant = gr.Dropdown(
                choices=["Essence", "Diesel", "Hybride", "Électrique"],
                label="Carburant", value="Essence"
            )
            transmission = gr.Radio(choices=["Manuelle", "Automatique"], value="Manuelle", label="Transmission")
            etat = gr.Dropdown(
                choices=["Mauvais", "Passable", "Bon", "Très bon", "Neuf"],
                value="Bon", label="État du véhicule"
            )
            consommation = gr.Slider(minimum=0.0, maximum=15.0, value=6.5, step=0.1, label="Consommation (L/100km)")

    gr.Markdown("---")

    btn = gr.Button("Estimer le prix", variant="primary", size="lg")

    prix_brut = gr.Number(label="Prix estimé (€)", precision=0, interactive=False)
    output = gr.Markdown(
        value="*Remplissez le formulaire et cliquez sur **Estimer le prix**...*",
        elem_id="result-box"
    )

    btn.click(
        fn=predire_prix,
        inputs=[marque, annee, kilometrage, puissance_cv,
                nb_portes, carburant, transmission,
                etat, nb_proprietaires, consommation],
        outputs=[prix_brut, output]
    )

    with gr.Tab("Prédiction Prix Carburant"):
        gr.Markdown("""
        # Prédiction des Prix Carburant à J+7
        **Modèle connecté au flux officiel des prix carburant.**
        Remplissez les informations ci-dessous pour obtenir une estimation.
        """)

        with gr.Row():
            departement = gr.Textbox(label="Département", placeholder="Exemple : 75")
            jour = gr.Slider(minimum=1, maximum=7, value=1, step=1, label="Jour (J+)")

        btn_carburant = gr.Button("Prédire", variant="primary")

        prix_carburant = gr.Textbox(label="Prix estimé (€)", interactive=False)
        details_carburant = gr.Markdown("*Cliquez sur Prédire pour voir les détails...*")

        btn_carburant.click(
            fn=predire_prix_carburant,
            inputs=[departement, jour],
            outputs=[prix_carburant, details_carburant]
        )

if __name__ == "__main__":
    demo.launch()
