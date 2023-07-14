import os
import io
import pickle

from flask import Flask, jsonify, request, Response

import shap

import pandas as pd

import matplotlib.pyplot as plt


app = Flask(__name__)

# Chargement des fichiers utiles dès le lancement
# de l'API pour que ce soit fait une seule fois

# Liste des colonnes utilisées pour la prédiction
with open("./data/columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Scaler de scikit pré-initialisé sur tout le dataset
with open("./data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Modèle de prédiction selectionné, entrainé sur
# tout le dataset préalablement scalé avec le scaler
with open("./data/model.pkl", "rb") as f:
    model = pickle.load(f)

# Explainer de SHAP pré-initialisé sur le modèle
with open("./data/explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# Seuil choisi
with open("./data/seuil.pkl", "rb") as f:
    seuil = pickle.load(f)


# Message de base pour vérifier quand l'API
# est en ligne
@app.route("/")
def index():
    return jsonify({"message": "hello world"})


# Fonction de prédiction de l'acceptation ou du refus d'un client
# Prends dans le GET une ligne client au format JSON, et retourne
# la prédiction au format binaire ainsi que la version probabilistique
@app.route("/predict", methods=["GET"])
def get_client_prediction():
    # Récupération des données
    client_line = request.json["data"]
    # Conversion de la ligne client reçu en une Series
    client_line = pd.read_json(client_line, typ="series")

    df = pd.to_numeric(client_line, errors="coerce")
    # ----- Recombinaison de certaines features ------
    client_line["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    client_line["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    client_line["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    client_line["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    client_line["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    client_line["EXT_SOURCE_MEAN_x_DAYS_EMPLOYED"] = (
        df["EXT_SOURCE_MEAN"] * df["DAYS_EMPLOYED"]
    )
    client_line["AMT_CREDIT_-_AMT_GOODS_PRICE"] = (
        df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    )
    client_line["AMT_CREDIT_r_AMT_GOODS_PRICE"] = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    )
    client_line["AMT_CREDIT_r_AMT_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    client_line["AMT_CREDIT_r_AMT_INCOME_TOTAL"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )
    client_line["AMT_INCOME_TOTAL_r_12_-_AMT_ANNUITY"] = (
        df["AMT_INCOME_TOTAL"] / 12.0 - df["AMT_ANNUITY"]
    )
    client_line["AMT_INCOME_TOTAL_r_AMT_ANNUITY"] = (
        df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    )
    client_line["CNT_CHILDREN_r_CNT_FAM_MEMBERS"] = (
        df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
    )

    # Scalling
    client_line = client_line[list(columns)].to_frame().T
    client_line_scaled = scaler.transform(client_line)
    # Lancement du modèle et récupératiob de la probabilité d'être refusé
    prediction = model.predict_proba(client_line_scaled)[0, 1]

    return jsonify(
        {
            "result": 1 if prediction > seuil else 0,
            "result_proba": prediction,
            "seuil": seuil,
        }
    )


# Fonction pour récupérer une ligne d'un client en JSON et retourne
# l'explication de la prédiction par SHAP
def explain_client(client_line):
    # Conversion de la ligne client reçu en une Series
    client_line = pd.read_json(client_line, typ="series")

    # Scalling de la ligne client
    client_line = client_line[list(columns)].to_frame().T
    client_line_scaled = scaler.transform(client_line)
    # Explication par SHAP du choix du modele
    client_line_explained = explainer(client_line_scaled)

    return client_line_explained


# Fonction de récupération des SHAP values depuis une ligne client
@app.route("/dataframe", methods=["GET"])
def get_client_dataframe():
    # Récupération des données et explication du choix avec SHAP
    client_line = request.json["data"]
    client_line_explained = explain_client(client_line)

    # Récupération des valeurs
    tmp = pd.DataFrame(client_line_explained.values, columns=columns, index=["shap"]).T
    # Ajout d'une version valeur absolut pour les classer par ordre d'importance
    tmp["abs"] = tmp["shap"].abs()
    tmp = tmp.sort_values("abs", ascending=False)

    return tmp.to_json()


# Fonction de génération de la waterfall de SHAP depuis une ligne client
@app.route("/plot/<forme>", methods=["GET"])
def get_client_plot(forme):
    # Récupération des données et explication du choix avec SHAP
    client_line = request.json["data"]
    client_line_explained = explain_client(client_line)

    # Initialisation d'une nouvelle figure
    plt.figure()
    # Génération de la waterfall
    shap.plots.waterfall(client_line_explained[0], max_display=10, show=False)
    # Récupération de la figure générées
    fig = plt.gcf()
    # Ajout d'indications sur la lecture du graphique
    plt.xticks([])
    plt.xlabel(
        "<"
        + "-" * 20
        + " " * 10
        + "Accepté"
        + " " * 40
        + "Refusé"
        + " " * 10
        + "-" * 20
        + ">"
    )
    # Ajustement de la taille (joue sur la taille du texte et le ratio)
    fig.set_size_inches(14, 6)
    fig.tight_layout()

    # Sauvegarder le graphique en tant que fichier image dans une mémoire tampon
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Renvoyer le contenu de la mémoire tampon comme réponse HTTP avec le type de contenu approprié
    return Response(buffer.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run()
