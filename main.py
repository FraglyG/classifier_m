from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

app = Flask(__name__)
model_path = os.environ.get('AI_MODEL_PATH')
classifier = pipeline("zero-shot-classification", model=model_path, device=0)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    sequence_to_classify = data['sequence']
    candidate_labels = data['labels']
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
