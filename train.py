import json
import nltk
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from model import ChatbotModel  # Importing the model class
from nltk.stem import WordNetLemmatizer
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Prepare data structures
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenize and preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for word in words:
        bag.append(1 if word in word_patterns else 0)  # Bag of Words

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1  # Assign class value

    training.append((bag, output_row))

# Convert training data to NumPy arrays
train_x = np.array([item[0] for item in training])  # Bag of words
train_y = np.array([item[1] for item in training])  # One-hot encoded outputs

# Model parameters
input_size = len(train_x[0])
hidden_size = 64  # Increased hidden layer size
output_size = len(classes)

# Initialize model
model = ChatbotModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  # Reduced learning rate and added weight decay

# Training loop
num_epochs = 500  # Increased number of epochs for better training
for epoch in range(num_epochs):
    inputs = torch.FloatTensor(train_x)
    labels = torch.LongTensor(np.argmax(train_y, axis=1))  # Convert one-hot to labels

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:  # Print loss every 50 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Load saved model, words, and classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize model and load saved state
input_size = len(words)
hidden_size = 64  # Same as during training
output_size = len(classes)
model = ChatbotModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()

def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Chat function
def chat():
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Convert input sentence to bag of words
        bow = bag_of_words(sentence, words)
        bow = torch.FloatTensor(bow).to(device)

        # Ensure that input is 2D (batch of 1)
        bow = bow.unsqueeze(0)

        # Predict class using the model
        output = model(bow)
        _, predicted = torch.max(output, dim=1)

        tag = classes[predicted.item()]

        # Find matching intent
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"Bot: {random.choice(intent['responses'])}")
                break

chat()