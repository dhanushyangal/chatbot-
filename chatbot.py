import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare data structures
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Add to words list
        documents.append((word_list, intent['tag']))  # Add to documents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add to classes

# Stemming and lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for word in words:
        bag.append(1 if word in word_patterns else 0)  # Create the Bag of Words

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1  # Assign 1 for the current class

    training.append((bag, output_row))  # Store as tuple

# Convert training to a NumPy array
train_x = np.array([item[0] for item in training])  # Extract bag of words
train_y = np.array([item[1] for item in training])  # Extract output rows

# Define the model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.2)  # Dropout layer for regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = len(train_x[0])  # Number of features (words)
hidden_size = 10  # Size of hidden layer (increased for complexity)
output_size = len(classes)  # Number of classes

# Initialize the model
model = ChatbotModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer for better convergence

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    inputs = torch.FloatTensor(train_x)  # Convert to tensor
    labels = torch.LongTensor(np.argmax(train_y, axis=1))  # Get labels from one-hot encoding

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')

# Function to convert user input to bag of words
def bag_of_words(s, words):
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    
    bag = [0] * len(words)
    for w in s_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Chat function
def chat():
    print("You can start chatting with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        # Process user input
        p = bag_of_words(user_input, words)
        p = torch.FloatTensor(p)
        
        # Predict the intent
        with torch.no_grad():
            output = model(p)
            _, predicted = torch.max(output, dim=0)
        
        intent_tag = classes[predicted.item()]
        
        # Get a random response for the predicted intent
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                break
        
        print(f"Bot: {response}")

# Run the chat function
if __name__ == "__main__":
    chat()
