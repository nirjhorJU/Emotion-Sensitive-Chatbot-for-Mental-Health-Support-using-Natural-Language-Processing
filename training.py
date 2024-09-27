import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import SGD
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Initialize variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents file
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Tokenize and process each pattern in the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Prepare the text data for TF-IDF
patterns = [' '.join(doc[0]) for doc in documents]
labels = [doc[1] for doc in documents]

# Use TF-IDF Vectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
train_x = tfidf_vectorizer.fit_transform(patterns).toarray()

# Save the TF-IDF vectorizer to use during prediction
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Prepare output for classification
output_empty = [0] * len(classes)
train_y = []

for doc in documents:
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    train_y.append(output_row)

train_y = np.array(train_y)

# Padding the sequences for LSTM compatibility
max_sequence_len = 1000
train_x = pad_sequences(train_x, maxlen=max_sequence_len)

# Create the LSTM-based model
model = Sequential()
model.add(Embedding(input_dim=train_x.shape[1], output_dim=128, input_length=max_sequence_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5')

print("Model created and saved")
