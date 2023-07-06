import os

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "hello world"})

@app.route('/api', methods=['GET'])
def search():
    args = request.args
    return jsonify(args)

if __name__ == '__main__':
    app.run()