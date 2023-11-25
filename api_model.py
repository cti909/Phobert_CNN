# -------------------------------------------
# run code: cmd ->  python api_model.py
# method: POST
# api: http://127.0.0.1:5000/api/model/predict-data
# post data: {"content": "sdasda"}
# reponse data (success):
# {
#     "data": {
#         "predict_code": 0,
#         "predict_message": "Real news"
#     },
#     "message": "Predict success",
#     "status": 200
# }
# reponse data (error):
# {
#     "error": "'content'",
#     "status": 500
# }
# -------------------------------------------
from flask import Flask, jsonify, request
from transformers import AutoTokenizer,AutoModel
import torch
from pyvi import ViTokenizer
import numpy as np
import re
import warnings
from keras.models import load_model

app = Flask(__name__)
app.secret_key = "secret_key"
app.config["SECRET_KEY"] = "super-secret-key"
warnings.filterwarnings("ignore")

with open('vietnamese.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

model_cnn_keras = load_model('cnn_model_100000.h5')
model_cnn_phobert = load_model('cnn_model_phobert.h5')
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

def preprocess(text):
    text = re.sub(r'\d+', ' ', text)  
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^`{|}~"""), ' ', text) 
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    text = text.strip()
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def tokenize_text(text):
    return ViTokenizer.tokenize(text)

def word_to_ids(text):
  input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')
  input_ids = np.array(input_ids, dtype=np.float32)
  return input_ids

def generate_sentence_embedding(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_ids)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

@app.route("/api/model/cnn-keras/predict", methods=['POST'])
def predict_cnn_keras():
    try:
        news_content = request.json['content']
        print(news_content)
        news_content = preprocess(news_content)
        news_content = tokenize_text(news_content)
        word_ids = np.array([word_to_ids(news_content)]) # (1,512)
        print(word_ids.shape)
        y_pred = model_cnn_keras.predict(word_ids)
        y_pred = y_pred.tolist()[0][0]
        predict_message = ""
        # y_pred in 0, 1
        if y_pred <= 0.5: 
            predict_message = "Real news" 
        else: 
            predict_message = "Fake news"
        reponse_data = {
            "status": 200,
            "message": "Predict success",
            "data": {
                "predict_code": y_pred,
                "predict_message": predict_message
            },
        }
        return jsonify(reponse_data)
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)})
    
@app.route("/api/model/cnn-phobert/predict", methods=['POST'])
def predict_cnn_phobert():
    try:
        news_content = request.json['content']
        print(news_content)
        news_content = preprocess(news_content)
        word_embedding = generate_sentence_embedding(news_content)
        y_pred = model_cnn_phobert.predict(np.array([word_embedding]))
        y_pred = y_pred.tolist()[0][0]
        print(y_pred)
        predict_message = ""
        # y_pred in 0, 1
        if y_pred <= 0.5: 
            predict_message = "Real news" 
        else: 
            predict_message = "Fake news"
        reponse_data = {
            "status": 200,
            "message": "Predict success",
            "data": {
                "predict_code": y_pred,
                "predict_message": predict_message
            },
        }
        return jsonify(reponse_data)
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)
