import nltk
import numpy as np
import torch
import torch.nn as nn
import random
import pickle

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch Normalization expects batch data

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if x.dim() > 1:  # Only apply batch normalization if input is a batch (2D or more)
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Function to convert user input to bag of words
def bag_of_words(s, words):
    s_words = nltk.word_tokenize(s)
    s_words = [nltk.stem.WordNetLemmatizer().lemmatize(word.lower()) for word in s_words]
    
    bag = [0] * len(words)
    for w in s_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Inference function
def get_response(user_input, model):
    p = bag_of_words(user_input, words)
    p = torch.FloatTensor(p)
    
    with torch.no_grad():
        output = model(p)
        _, predicted = torch.max(output, dim=0)
    
    intent_tag = classes[predicted.item()]
    return intent_tag
