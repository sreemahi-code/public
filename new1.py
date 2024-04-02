import random
import json
import pickle
import numpy as np
import nltk

from ntlk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer(0)

intents = json.loads(open('inents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model("chatbot
