import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data and model
lemmatizer = WordNetLemmatizer()

# Load TF-IDF vectorizer and other necessary files
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = load_model('model.h5')

# Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Function to preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return ' '.join(sentence_words)

# Function to generate TF-IDF features
def tfidf_features(sentence):
    sentence = clean_up_sentence(sentence)
    sentence_tfidf = tfidf_vectorizer.transform([sentence]).toarray()
    return pad_sequences(sentence_tfidf, maxlen=max_sequence_len)

# Function to predict class
def predict_class(sentence):
    x_test = tfidf_features(sentence)
    prediction = model.predict(x_test)
    predicted_class = classes[np.argmax(prediction[0])]
    return predicted_class

# Function to get full response
def get_response(sentence):
    predicted_class = predict_class(sentence)
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response
    return "I'm not sure I understand. Can you rephrase that?"

# Main chat loop
print("I am an emotion-sensitive chatbot for mental health support. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    bot_response = get_response(user_input)
    print("Chatbot:", bot_response)
