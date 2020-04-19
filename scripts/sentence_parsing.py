import numpy as np
import pandas as pd


def tag_sentence(input, model, word2idx):
    sequence = np.ones((1, 50)).astype(int) * 25947
    if isinstance(input[0], str):
        sentence = input.split(' ')
        for i, key in enumerate(sentence):
            try:
                sequence[:, i] = word2idx[key]
            except:
                print("key {0} is unknown".format(key))
                sequence[:, i] = 25947
    else:
        for i, key in enumerate(input):
            sequence[:, i] = key

    output = np.argmax(model.predict(sequence), axis=2)
    return output


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class SentenceGetterNEROnly(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                     s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def get_data(filepath, encoding, task="NER"):
    if encoding == "utf-8":
        data = pd.read_csv(filepath, encoding="utf-8", engine='python', quotechar="^", error_bad_lines=False)
    else:
        data = pd.read_csv(filepath, encoding="latin1", engine='python', quotechar="^", error_bad_lines=False)

    data = data.fillna(method="ffill")
    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    if task == "NER":
        tags = list(set(data["Tag"].values))
    else:
        tags = list(set(data["POS"].values))
    n_tags = len(tags)
    return data, n_words, tags, n_tags


def get_words(data):
    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    return words