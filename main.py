from posixpath import dirname
from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
model_path = os.environ.get('AI_MODEL_PATH')
full_model_path = (f'{dirname(__file__)}{model_path}')
classifier = pipeline("zero-shot-classification", model=full_model_path)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    sequence_to_classify = data['sequence']
    candidate_labels = data['labels']
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return jsonify(output)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='::', port=port, debug=True)