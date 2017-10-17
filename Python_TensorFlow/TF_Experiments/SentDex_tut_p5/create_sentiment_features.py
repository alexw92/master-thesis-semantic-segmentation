# [char, table, spoon, television]
#
# I pulled the charir up to the table
#
# [1 1 0 0]

# Used commands
# CMD:
# pip3 install nltk
# Python:
# import nltk
# nltk.download()
# d  all  (or click in GUI on 'all')

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readline()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    #  w_counts = {'the':52521, 'and':25242}

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    # [
    # [[0 1 0 1 1 0], [0 1]]
    #
    # ]
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize((i)) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [1, 0])
    random.shuffle(features)

    # does tf.argmax([output])  == tf.argmax([expectations]) e.g. was exp right?
    # https://www.youtube.com/watch?v=YFxVHD2TNII 14:22
