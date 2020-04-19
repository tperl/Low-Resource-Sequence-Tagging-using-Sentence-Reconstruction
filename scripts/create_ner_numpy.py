from sentence_parsing import SentenceGetter, get_data, get_words
import pandas as pd
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

import json


def main_run(lang):
    print('generating for language =',lang)

    if lang == "dutch":
        train_path = r"../../data/Conll2002/ned/ned.train_updated.csv"
        dev_path = r"../../data/Conll2002/ned/ned.testa_updated.csv"
        test_path = r"../../data/Conll2002/ned/ned.testb_updated.csv"
    elif lang == "spanish":
        train_path = r"../../data//Conll2002/span/esp.train_updated.csv"
        dev_path = r"../../data//Conll2002/span/esp.testa_updated.csv"
        test_path = r"../../data/Conll2002/span/esp.testb_updated.csv"
    else: #english
        train_path = r"../../data/conLL2003/eng/eng.train_updated.csv"
        dev_path = r"../../data/conLL2003/eng/eng.testa_updated.csv"
        test_path = r"../../data/conLL2003/eng/eng.testb_updated.csv"

    if lang == 'english' or lang == 'genia':
        encoding = "latin1"
    else:
        encoding = "utf-8"
    # get data and tags
    data, n_words, tags, n_tags = get_data(train_path,encoding)
    data_test, n_words_test, tags_test, n_tags_test = get_data(test_path, encoding)
    data_dev, n_words_dev, tags_dev, n_tags_dev = get_data(dev_path, encoding)

    if lang == 'genia':
        tags.append('O')
        tags.append('IN|CC')
        tags.append('VBP|VBZ')
        n_tags += 3
    # constructor for sentence getter
    getter = SentenceGetter(data)
    getter_test = SentenceGetter(data_test)
    getter_dev = SentenceGetter(data_dev)

    # get sentences for training and testing
    sentences = getter.sentences
    sentences_test = getter_test.sentences
    sentences_dev = getter_dev.sentences
    len_list = [len(s) for s in sentences]

    words = get_words(data)
    words_test = get_words(data_test)
    words_dev = get_words(data_dev)

    words = list(set(words+words_test+words_dev))
    words.append("PAD")
    n_words = len(words)
    max_len = max(len_list)
    max_len = 50
    word2idx = {w: i for i, w in enumerate(words)}
    # tag2idx = {t: i for i, t in enumerate(tags)}
    tag2idx = {'B-PER': 0, 'S-LOC': 1, 'S-ORG': 2, 'S-PER': 3, 'B-LOC': 4, 'E-LOC': 5, 'E-MISC': 6, 'B-ORG': 7, 'I-ORG': 8, 'S-MISC': 9, 'E-PER': 10, 'I-LOC': 11, 'B-MISC': 12, 'I-PER': 13, 'O': 14, 'I-MISC': 15, 'E-ORG': 16}

    if 'O' not in tag2idx:
        tag2idx['O'] = len(tag2idx)
        n_tags += 1

    word_len_list = [len(word) for word in words]
    max_len_char = 20
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)

    # char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx_english = {'A': 2, 'k': 3, 'h': 4, 'v': 5, 'u': 6, '&': 7, 'n': 8, 'g': 9, '@': 10, 'j': 11, 'U': 12, 'F': 13, 'P': 14, '5': 15, 'N': 16, 'T': 17, 'I': 18, 'p': 19, 'M': 20, '%': 21, ':': 22, '`': 23, ']': 24, '=': 25, 'c': 26, 'w': 27, 'H': 28, 'y': 29, '[': 30, '7': 31, 'L': 32, '0': 33, '9': 34, 'V': 35, 'W': 36, 'f': 37, 't': 38, '1': 39, 'G': 40, '/': 41, '*': 42, 'B': 43, 'q': 44, 'Z': 45, 'K': 46, 'C': 47, 'E': 48, '6': 49, 'Y': 50, ',': 51, '-': 52, '2': 53, 'x': 54, ';': 55, 'S': 56, 'O': 57, 'r': 58, '"': 59, '#': 60, ')': 61, 'D': 62, 's': 63, "'": 64, 'i': 65, 'e': 66, '.': 67, '!': 68, 'X': 69, '?': 70, 'b': 71, '$': 72, 'a': 73, 'J': 74, 'd': 75, 'z': 76, 'm': 77, 'o': 78, '+': 79, '4': 80, '3': 81, 'Q': 82, 'l': 83, '8': 84, '(': 85, 'R': 86}
    char2idx_spanish = {'w': 2, 'O': 3, '-': 4, 'n': 5, 'Z': 6, 'r': 7, 'B': 8, 'L': 9, 'e': 10, 'V': 11, 'H': 12, 'ú': 13, 'b': 14, 'f': 15, 'í': 16, '"': 17, 'Q': 18, 'c': 19, '?': 20, '9': 21, '·': 22, '/': 23, '4': 24, '2': 25, 'u': 26, 'Ñ': 27, 'é': 28, 't': 29, 'E': 30, 'p': 31, 'F': 32, 'h': 33, '7': 34, '1': 35, 'g': 36, 'U': 37, 'J': 38, 'i': 39, ';': 40, '6': 41, 'R': 42, '+': 43, '(': 44, ',': 45, 'A': 46, 'M': 47, '.': 48, 'o': 49, ')': 50, '=': 51, 'X': 52, '&': 53, 'j': 54, 'ü': 55, '%': 56, 'T': 57, 'q': 58, 'z': 59, 'k': 60, "'": 61, 'l': 62, 'I': 63, 'm': 64, 'ó': 65, '0': 66, 'G': 67, 'P': 68, 'd': 69, '8': 70, '5': 71, '!': 72, '@': 73, 'C': 74, 'á': 75, 'W': 76, 's': 77, 'v': 78, 'D': 79, 'y': 80, 'N': 81, '3': 82, 'S': 83, 'Y': 84, 'a': 85, 'K': 86, 'à': 87, 'ñ': 88, '¿': 89, ':': 90, '¡': 91, 'x': 92}
    char2idx_dutch = {'O': 2, '(': 3, 'x': 4, ':': 5, 'e': 6, 'v': 7, 's': 8, ' ': 9, 'ô': 10, 'ê': 11, 'q': 12, 'í': 13, 'y': 14, 'L': 15, 'N': 16, 'ä': 17, 'à': 18, 'K': 19, '6': 20, 'î': 21, 'è': 22, 'X': 23, '-': 24, 'M': 25, 'f': 26, 'h': 27, 'c': 28, '.': 29, ',': 30, 'g': 31, 'i': 32, 'ñ': 33, 'ö': 34, 'w': 35, 'A': 36, 'H': 37, '1': 38, '9': 39, '8': 40, 'Z': 41, 'n': 42, 'z': 43, 'E': 44, 'á': 45, 'R': 46, 'Å': 47, 'o': 48, 'Q': 49, 'W': 50, 'B': 51, '0': 52, '?': 53, '2': 54, 'u': 55, 'r': 56, '"': 57, "'": 58, 'd': 59, 'G': 60, '5': 61, 'F': 62, '+': 63, 'V': 64, 'Y': 65, 'C': 66, 'p': 67, '@': 68, 't': 69, 'J': 70, 'ë': 71, '°': 72, 'a': 73, 'Ö': 74, ')': 75, 'â': 76, 'ã': 77, 'j': 78, 'U': 79, 'ú': 80, 'm': 81, '©': 82, '7': 83, 'D': 84, 'é': 85, 'ç': 86, 'ü': 87, 'ó': 88, 'I': 89, '_': 90, '&': 91, 'b': 92, 'T': 93, '3': 94, 'P': 95, '=': 96, 'S': 97, '%': 98, 'ï': 99, '4': 100, '/': 101, 'l': 102, '!': 103, 'k': 104}
    char2idx = {**char2idx_english, **char2idx_spanish, **char2idx_dutch}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    # pack and split the data for training
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    X_tr, y_tr = X,y
    X_char_tr = X_char

    # Prepare Test data
    def getTestData(sentences):
        X_test = []
        for s in sentences:
            sen = []
            for w in s:
                try:
                    ii = word2idx[w[0]]
                except KeyError:
                    ii = n_words -1
                sen.append(ii)
            X_test.append(sen)

        X_char_test = []
        for sentence in sentences:
            sent_seq = []
            for i in range(max_len):
                word_seq = []
                for j in range(max_len_char):
                    try:
                        word_seq.append(char2idx.get(sentence[i][0][j]))
                    except:
                        word_seq.append(char2idx.get("PAD"))
                sent_seq.append(word_seq)
            X_char_test.append(np.array(sent_seq))

        X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=n_words-1)
        y_t = [[tag2idx[w[2]] for w in s] for s in sentences]
        y_t = pad_sequences(maxlen=max_len, sequences=y_t, padding="post", value=tag2idx["O"])
        y_t = [to_categorical(i, num_classes=n_tags) for i in y_t]

        return X_test,X_char_test, y_t

    X_test, X_char_test, y_test = getTestData(sentences_test)
    X_dev, X_char_dev, y_dev = getTestData(sentences_dev)

    use_muse = True
    if use_muse:
        if lang == 'dutch' or lang == 'dutch_ud':
            embedding_path = '../../data/embeddings/MUSE/wiki.multi.nl.vec'
        elif lang == 'spanish':
            embedding_path = '../../data/embeddings/MUSE/wiki.multi.es.vec'
        else: # english
            embedding_path ='../../data/embeddings/MUSE/wiki.multi.en.vec'
    else:
        # embedding_path = "I:/Documents/TAL MSC/nlp/gitRepo/tranfser-learning-for-sequence-tagging/cleaner_project2/word2vec_embeddings/eswiki.cbow.50d.txt"
        embedding_path = 'I:/Documents/TAL MSC/nlp/glove.6B/glove.6B.100d.txt'
    esp_aligned_emb = pd.read_csv(embedding_path, delimiter='  ',
                                  encoding="utf-8")

    emb_size = 100
    if use_muse:
        emb_size = 300

    esp_emb_len = len(esp_aligned_emb.values)
    esp_emb_dict = {}
    esp_emb_dim = len(esp_aligned_emb.values[0, 0].split(' ')[1:])
    esp_embedding_matrix = np.zeros((esp_emb_len + 1, emb_size))

    for i in range(esp_emb_len):
        key = esp_aligned_emb.values[i, 0].split(' ')[0]
        embedding = np.zeros(len(esp_aligned_emb.values[i, 0].split(' ')) - 1)
        for j, emb in enumerate(esp_aligned_emb.values[i, 0].split(' ')[1:]):
            embedding[j] = emb

        if embedding.shape[0] < emb_size:
            esp_embedding_matrix[i + 1][0:emb_size - 1] = embedding
            esp_embedding_matrix[i + 1][emb_size - 1] = np.random.rand() - 0.5
        else:
            esp_embedding_matrix[i + 1] = embedding
        esp_emb_dict[key] = i + 1


    X_esp_muse_tr = np.zeros(X_tr.shape)


    for i in range(X_tr.shape[0]):
        for j in range(X_tr.shape[1]):
            # if words[X_tr[i][j]] == "PAD":
            if False:
                X_esp_muse_tr[i][j] = 0
            else:
                try:
                    X_esp_muse_tr[i][j] = esp_emb_dict[words[X_tr[i][j]].lower()]
                except KeyError:
                    X_esp_muse_tr[i][j] = esp_emb_dict['unknown']

    def getMuseX(X_test):
        X_esp_muse_test = np.zeros(X_test.shape)
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                # if words[X_test[i][j]] == "PAD":
                if False:
                    X_esp_muse_test[i][j] = 0
                else:
                    try:
                        X_esp_muse_test[i][j] = esp_emb_dict[words[X_test[i][j]].lower()]
                    except KeyError:
                        X_esp_muse_test[i][j] = esp_emb_dict['unknown']

        return X_esp_muse_test

    X_esp_muse_test = getMuseX(X_test)
    X_esp_muse_dev = getMuseX(X_dev)

    idx2tag = {}
    for key in tag2idx.keys():
            idx2tag[tag2idx[key]] = key
    idx2char = {}
    for key in char2idx.keys():
            idx2char[char2idx[key]] = key
    esp_emb_dict
    idx2word = {}
    for key in esp_emb_dict.keys():
        idx2word[esp_emb_dict[key]] = key

    print(idx2tag)
    np.save('../../data/parsed/'+lang+'/X_muse',X_esp_muse_tr)
    np.save('../../data/parsed/'+lang+'/X_char',X_char_tr)
    np.save('../../data/parsed/'+lang+'/y',y)
    np.save('../../data/parsed/'+lang+'/X_muse_test',X_esp_muse_test)
    np.save('../../data/parsed/'+lang+'/X_char_test',X_char_test)
    np.save('../../data/parsed/'+lang+'/y_test',np.array(y_test))
    np.save('../../data/parsed/' + lang + '/X_muse_dev', X_esp_muse_dev)
    np.save('../../data/parsed/' + lang + '/X_char_dev', X_char_dev)
    np.save('../../data/parsed/' + lang + '/y_dev', np.array(y_dev))
    np.save('../../data/parsed/'+lang+'/embedding_matrix',esp_embedding_matrix)

    with open('../../data/parsed/' + lang + '/idx2word.json','w') as f:
        json.dump(idx2word, f,indent=4, sort_keys=True)
    with open('../../data/parsed/' + lang + '/idx2tag.json','w') as f:
        json.dump(idx2tag, f, indent=4, sort_keys=True)
    with open('../../data/parsed/' + lang + '/idx2char.json','w',encoding='utf-8') as f:
        json.dump(idx2char, f, indent=4, sort_keys=True)
    print('numpy arrays generated for lang =',lang)

def main():
    print('generating numpy array for training...')
    main_run('english')
    main_run('spanish')
    main_run('dutch')

if __name__ == "__main__":
    main()
    print("Done")