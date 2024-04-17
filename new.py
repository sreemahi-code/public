import random
import json
import pickle
import numpy as np
import tensorflow as tf 

import nltk 
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

file_path = '/Users/anuradhagangadharan/Desktop/chatbott/chatbott/include/intents.json'  # Ensure the filename is included

# Use a with statement to handle the file opening and closing
with open(file_path, 'r') as file:
    intents = json.load(file)
##intents = json.loads(open(/Users/anuradhagangadharan/Desktop/chatbott/chatbott/include/intents.json')).read())

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
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPattern = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower())for word in wordPattern]
    for word in words: bag.append(1) if word in wordPattern else bag.append(0)

    outputRow = list(outputEmpty) 
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[: , len(words):]
trainY = training [ :,  len(words):]


model = tf.keras.Sequential()


'''model.add(tf.keras.layers.Dense(128, input_shape=(23,), activation='relu'))  # Adjust the input_shape to match the feature size of your input data
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(classes), activation='softmax'))  # Assuming `classes` is the number of output classes

# Compile and train the model as usual

#model.add(tf.keras.layers.Dense(128, input_shape= (len(trainX[0]), ),activation = 'relu'))
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(64, activation = 'relu'))
#model.add(tf.keras.layers.Dense (len(trainY[0]), activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1) '''


input_shape = trainX.shape[1]
model.add(tf.keras.layers.Dense(128, input_shape=(input_shape,), activation='relu'))  # Adjust input_shape here
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_m1.h5', hist)
print("executed babe")


