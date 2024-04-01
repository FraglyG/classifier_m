from flask import Flask, request, jsonify
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

app = Flask(__name__)
model_path = os.environ.get('AI_MODEL_PATH')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=False)

classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    sequence_to_classify = data['sequence']
    candidate_labels = data['labels']
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
