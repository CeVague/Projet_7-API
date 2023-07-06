import os
import pickle

from flask import Flask, jsonify, request

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

@app.route('/api', methods=['GET'])
def get_client_shap():
    client_line = data = request.json['data']
    client_line = pd.read_json(client_line, typ='series')
    
    client_line_scaled = scaler.transform([client_line[list(columns)]])
    client_line_explained = explainer(client_line_scaled)
    
    tmp = pd.DataFrame(client_line_explained.values, columns=columns, index=['shap']).T
    tmp['abs'] = tmp['shap'].abs()
    tmp = tmp.sort_values('abs', ascending=False)
    return tmp.to_json()

if __name__ == '__main__':
    app.run()