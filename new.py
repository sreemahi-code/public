import random
import json
import pickle
import numpy as np
import tensorflow as tf 

import nltk 
from nltk.stem import WorldNetLemmatizer

lemmatizer = WorldNetLemmatizer()

intents = json.loads(open('intents.json').read())

words  = []
classes = []
documents = []
ignoreLetters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
    wordList = nltk.word_tokenize(pattern)
    words.extend(wordList)
    documents.append((wordList, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
    
words = [lemmatizer.lemmatize(word)for word in words if word not in ignoreLetters]
words = sorted(set(classes))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPattern = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower())for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPattern else bag.append(0)

    outputRow = list(outputEmpty) 
    outputRow[classes.indexdocument[1]] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[: , :len(words)]
trainY = traning [ ;, : len(words)]


model = tf.keras.Sequential()

model.add(tf.keras.layer.Dense(128, input_shape(len(trainX[0]), ),activation = 'relu'))
model.add(tf.keras.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense (len(trainY[0], activation = 'softmax'))
          
