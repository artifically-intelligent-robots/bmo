import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json

with open("quotes.json") as file:
    data = json.load(file)

# all stemmed words in all patterns
words = []
# all tag labels
labels = []
# list of all the different patterns
pattern_x = []
# corresponding 'tag' label (to pattern_x)
tag_y = []

#load in json file data
for intent in data["intents"]:
    # for each input pattern
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        pattern_x.append(tokens)
        tag_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stem all words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]

#remove duplicate words, convert back into list, sort it alphabetic
words = sorted(list(set(words)))


training = []
output = []
# empty list size of labels
out_empty = [0 for _ in range(len(labels))]

# one hot encoded words in patterns_x
# create a bag of words
for i, pattern in enumerate(pattern_x):
    bag = []

    # all stemmed words in THIS pattern
    tokens = [stemmer.stem(w) for w in pattern]

    # 1 = word exists, 0 = word does not exist in pattern
    for w in words:
        if w in tokens:
            bag.append(1)
        else:
            bag.append(0)

    # copy over empty array size of labels
    output_row = out_empty[:]

    # ex.
    # output row [index of (tag idx of current pattern)]
    # tag_y[1] = Goodbye
    # labels.index(goodbye) = 3 ; labels [greeting, anger, sad, goodbye]
    # output_row[3] = 1
    # this tells which tag correlates with pattern
    output_row[labels.index(tag_y[i])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)
''' Preprocessing complete '''


tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")












#
