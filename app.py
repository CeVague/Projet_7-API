import os
import io
import pickle

from flask import Flask, jsonify, request, Response
import shap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


import pandas as pd

app = Flask(__name__)

with open('./data/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)
with open('./data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./data/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./data/columns.pkl', 'rb') as f:
    columns = pickle.load(f)

@app.route('/')
def index():
    return jsonify({"message": "hello world"})

def explain_client(client_line):
    client_line = pd.read_json(client_line, typ='series')
    
    client_line_scaled = scaler.transform([client_line[list(columns)]])
    client_line_explained = explainer(client_line_scaled)
    
    return client_line_explained

@app.route('/dataframe', methods=['GET'])
def get_client_dataframe():
    client_line = request.json['data']
    client_line_explained = explain_client(client_line)
    
    tmp = pd.DataFrame(client_line_explained.values, columns=columns, index=['shap']).T
    tmp['abs'] = tmp['shap'].abs()
    tmp = tmp.sort_values('abs', ascending=False)
    return tmp.to_json()

@app.route('/plot/<forme>', methods=['GET'])
def get_client_plot(forme):
    client_line = request.json['data']
    client_line_explained = explain_client(client_line)
    
    # Créer le graphique avec Matplotlib
    plt.figure()
    shap.plots.waterfall(client_line_explained[0], max_display=10, show=False)
    fig = plt.gcf()
    plt.xticks([])
    plt.xlabel("<"+"-"*20+" "*10+"Accepté"+" "*40+"Refusé"+" "*10+"-"*20+">")
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