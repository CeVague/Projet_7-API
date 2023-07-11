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
with open('./data/columns.pkl', 'rb') as f:
    columns = pickle.load(f)
    
# Scaler de scikit pré-initialisé sur tout le dataset
with open('./data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Modèle de prédiction selectionné, entrainé sur
# tout le dataset préalablement scalé avec le scaler
with open('./data/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Explainer de SHAP pré-initialisé sur le modèle
with open('./data/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)
    
    
# Message de base pour vérifier quand l'API
# est en ligne
@app.route('/')
def index():
    return jsonify({"message": "hello world"})

def is_number(s):
    try:
        float(s)
        return True
    except :
        return False

def preprocess_client(df):
    dff = pd.Series(index=df.index)
    
    for i, v in df.items():
        if is_number(v):
            dff[i] = float(v)
    
    df["DAYS_EMPLOYED_PERC"] = dff["DAYS_EMPLOYED"] / dff["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = dff["AMT_INCOME_TOTAL"] / dff["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = dff["AMT_INCOME_TOTAL"] / dff["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = dff["AMT_ANNUITY"] / dff["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = dff["AMT_ANNUITY"] / dff["AMT_CREDIT"]
    
    df["EXT_SOURCE_MEAN_x_DAYS_EMPLOYED"] = dff["EXT_SOURCE_MEAN"] * dff["DAYS_EMPLOYED"]
    df["AMT_CREDIT_-_AMT_GOODS_PRICE"] = dff["AMT_CREDIT"] - dff["AMT_GOODS_PRICE"]
    df["AMT_CREDIT_r_AMT_GOODS_PRICE"] = dff["AMT_CREDIT"] / dff["AMT_GOODS_PRICE"]
    df["AMT_CREDIT_r_AMT_ANNUITY"] = dff["AMT_CREDIT"] / dff["AMT_ANNUITY"]
    df["AMT_CREDIT_r_AMT_INCOME_TOTAL"] = dff["AMT_CREDIT"] / dff["AMT_INCOME_TOTAL"]
    df["AMT_INCOME_TOTAL_r_12_-_AMT_ANNUITY"] = (
        dff["AMT_INCOME_TOTAL"] / 12.0 - dff["AMT_ANNUITY"]
    )
    df["AMT_INCOME_TOTAL_r_AMT_ANNUITY"] = dff["AMT_INCOME_TOTAL"] / dff["AMT_ANNUITY"]
    df["CNT_CHILDREN_r_CNT_FAM_MEMBERS"] = dff["CNT_CHILDREN"] / dff["CNT_FAM_MEMBERS"]
    
    return df

# Fonction de prédiction de l'acceptation ou du refus d'un client
# Prends dans le GET une ligne client au format JSON, et retourne 
# la prédiction au format binaire ainsi que la version probabilistique
@app.route('/predict', methods=['GET'])
def get_client_prediction():
    # Récupération des données
    client_line = request.json['data']
    # Conversion de la ligne client reçu en une Series
    client_line = pd.read_json(client_line, typ='series')
    
    client_line = preprocess_client(client_line)
    
    # Scalling
    client_line_scaled = scaler.transform([client_line[list(columns)]])
    # Lancement du modèle et récupératiob de la probabilité d'être refusé
    prediction = model.predict_proba(client_line_scaled)[0, 1]
    
    return jsonify({"result": 1 if prediction>0.09 else 0, "result_proba": prediction})

# Fonction pour récupérer une ligne d'un client en JSON et retourne
# l'explication de la prédiction par SHAP
def explain_client(client_line):
    # Conversion de la ligne client reçu en une Series
    client_line = pd.read_json(client_line, typ='series')
    
    # Scalling de la ligne client
    client_line_scaled = scaler.transform([client_line[list(columns)]])
    # Explication par SHAP du choix du modele
    client_line_explained = explainer(client_line_scaled)
    
    return client_line_explained

# Fonction de récupération des SHAP values depuis une ligne client
@app.route('/dataframe', methods=['GET'])
def get_client_dataframe():
    # Récupération des données et explication du choix avec SHAP
    client_line = request.json['data']
    client_line_explained = explain_client(client_line)
    
    # Récupération des valeurs
    tmp = pd.DataFrame(client_line_explained.values, columns=columns, index=['shap']).T
    # Ajout d'une version valeur absolut pour les classer par ordre d'importance
    tmp['abs'] = tmp['shap'].abs()
    tmp = tmp.sort_values('abs', ascending=False)
    
    return tmp.to_json()

# Fonction de génération de la waterfall de SHAP depuis une ligne client
@app.route('/plot/<forme>', methods=['GET'])
def get_client_plot(forme):
    # Récupération des données et explication du choix avec SHAP
    client_line = request.json['data']
    client_line_explained = explain_client(client_line)
    
    # Initialisation d'une nouvelle figure
    plt.figure()
    # Génération de la waterfall
    shap.plots.waterfall(client_line_explained[0], max_display=10, show=False)
    # Récupération de la figure générées
    fig = plt.gcf()
    # Ajout d'indications sur la lecture du graphique
    plt.xticks([])
    plt.xlabel("<"+"-"*20+" "*10+"Accepté"+" "*40+"Refusé"+" "*10+"-"*20+">")
    # Ajustement de la taille (joue sur la taille du texte et le ratio)
    fig.set_size_inches(14, 6)
    fig.tight_layout()

    # Sauvegarder le graphique en tant que fichier image dans une mémoire tampon
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Renvoyer le contenu de la mémoire tampon comme réponse HTTP avec le type de contenu approprié
    return Response(buffer.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run()