import pickle
from sre_parse import Tokenizer
import tokenize
from cgitb import text
from fileinput import filename

import joblib
import pandas as pd
# from sklearn.externals import joblib
import sklearn.externals as joblib
from flask import Flask, render_template, request, url_for
from pyexpat import model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import T5Tokenizer as tokenizer
from transformers import T5Tokenizer,T5ForConditionalGeneration,Adafactor

# load the model from disk
# filename = 'nlp_model.pkl'
filename = '2text_summarization_model.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
app = Flask(__name__)

import transformers 
print(transformers.__version__)

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer)

#tokenizer_loader = Tokenizer
# model_loader = AutoModelForSequenceClassification

MODEL_NAME = 't5-base'
# instantiate the tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# print(tokenizer)


def summarizeText(text): 
    text_encoding = tokenize(
    text, max_length=512, padding='max_length',
    truncation=True, 
    return_attention_mask=True, 
    add_special_tokens=True, 
    return_tensors='pt')
    
    generated_ids = model.generate(input_ids=text_encoding['input_ids'], attention_mask=text_encoding['attention_mask'],
                               max_length=128, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
    preds = [
    tokenizer.decode(gen_id, skip_special_tokens=True,
                     clean_up_tokenization_spaces=True)
    for gen_id in generated_ids]
    return "   ".join(preds)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/summarizeText', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = [message]
        output = summarizeText(text)
    return render_template('result.html', prediction=output)


if __name__ == '__main__':
    app.run(debug=True)
