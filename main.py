from posixpath import dirname
from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

# adding this comment cuz it didn't detect my previous commit
app = Flask(__name__)
model_path = os.environ.get('AI_MODEL_PATH')
full_model_path = (f'{dirname(__file__)}{model_path}')
classifier = pipeline("zero-shot-classification", model=full_model_path)

# second AI model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(full_model_path)
model = AutoModelForSequenceClassification.from_pretrained(full_model_path)

# classifyv2

@app.route('/classifyv2', methods=['POST'])
def classifyv2():
    data = request.get_json()
    sequence_to_classify = data['sequence']
    context = data['context']
    candidate_labels = data['labels']

    input = tokenizer(sequence_to_classify, context, return_tensors="pt") #truncation=True
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, candidate_labels)}
    print(prediction)
    return jsonify(prediction)


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
    app.run(host='::', port=5000, debug=True)