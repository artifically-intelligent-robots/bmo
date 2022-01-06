import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer

import numpy
import tflearn
import tensorflow
import random
import json

with open("quotes.json") as file:
    data = json.load(file)


words = []
labels = []
docs_pattern = []
docs_tag = []

#preprocessing data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_pattern.append(pattern)
        docs_tag.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
}

# stem all words
words = [stemmer.stem(w.lower()) for w in words]

#remove duplicate words, convert back into list, sort it alphabetic
words = sorted(list(set(words)))
