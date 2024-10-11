from flask import Flask, render_template, request, jsonify
import torch
from model import ChatbotModel
import nltk
import numpy as np
import random
import json
import pickle

# Load the model and other necessary data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load intents and model files
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
input_size = len(words)
hidden_size = 64
output_size = len(classes)
model = ChatbotModel(input_size, hidden_size, output_size).to(device)

# Load only the weights (recommended for security)
model.load_state_dict(torch.load('chatbot_model.pth', weights_only=True))
model.eval()


# Flask app initialization
app = Flask(__name__)

# Function to preprocess user input
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Function to get the chatbot response
def get_response(message):
    bow = bag_of_words(message, words)
    bow = torch.FloatTensor(bow).unsqueeze(0).to(device)
    
    output = model(bow)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    for intent in intents['intents']:
        if tag == intent['tag']:
            return random.choice(intent['responses'])

    return "I don't understand. Could you please rephrase?"

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chatbot responses
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    data = request.get_json()
    user_message = data['message']
    bot_response = get_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
