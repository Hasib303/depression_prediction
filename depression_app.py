from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('depression_model.h5')
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

max_sequence_length = model.input_shape[1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    prediction = model.predict(padded_sequence)[0][0]
    prediction_percentage = prediction * 100
    
    return jsonify({'prediction': f'{prediction_percentage:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)
