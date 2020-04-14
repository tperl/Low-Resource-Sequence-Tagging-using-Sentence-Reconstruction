import argparse
import numpy as np
from keras.optimizers import Adam
from keras.models import Input
from utils.f1_test import f1_test
import json
import os


def run_test(X_muse_test, X_char_test, idx2tag, model, max_len, y_test, debug_mode=False):
    pred_tags = []
    for i in range(0,int(len(X_muse_test)/100)):
        preds = np.argmax(model.predict_on_batch([X_muse_test[i*100:(i+1)*100], np.array(X_char_test[i*100:(i+1)*100])]), axis=2)

        for p in preds:
            pred_tags.append([idx2tag[str(p[j])] for j in range(p.shape[0])])

    i+=1
    i = i*100
    while i <len(X_muse_test):
        preds = np.argmax(model.predict_on_batch([X_muse_test[i * 1:(i + 1) * 1], np.array(X_char_test[i * 1:(i + 1) * 1])]),axis=2)[0]
        pred_tags.append([idx2tag[str(preds[j])] for j in range(len(preds))])
        i+= 1
    gt_tags = np.argmax(y_test,axis=2)
    gt_tags = [[idx2tag[str(gt_tags[j][i])] for i in range(max_len)] for j in range(X_muse_test.shape[0])]
    f1_test(gt_tags,pred_tags, debug_mode)


def main(parser):
    args = parser.parse_args()
    lang = args.lang

    with open('../data/parsed/' + lang + '/idx2tag.json') as f:
        idx2tag = json.load(f)

    X_tgt_muse_test = np.load('../data/parsed/' + lang + '/X_muse_test.npy')
    X_char_test = np.load('../data/parsed/' + lang + '/X_char_test.npy')
    y_test = np.load('../data/parsed/' + lang + '/y_test.npy')
    embedding_matrix = np.load('../data/parsed/' + lang + '/embedding_matrix.npy')

    # Vars for each corpus
    n_chars = np.max(X_char_test) - 1
    max_len = X_tgt_muse_test.shape[-1]
    max_len_char = X_char_test.shape[-1]
    n_tags = y_test.shape[-1]

    if args.model == 'baseline':
        from models.baseline import BaselineModel
        word_in = Input(shape=(max_len,))
        char_in = Input(shape=(max_len, max_len_char,))
        h1 = 70
        h2 = 70
        h3 = 140
        output_dim = 70
        model, crf = BaselineModel(word_in, char_in, embedding_matrix, n_chars, output_dim, max_len_char, max_len, h1, h2, h3, n_tags)

    if args.model == 'baseline':
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        model.load_weights('saved_models/model_baseline.h5')

    run_test(X_tgt_muse_test, X_char_test, idx2tag, model, max_len, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='What Language to train',default='english')
    parser.add_argument('--model', help='Flag for what model to choose baseline/reconstruction/transferlearning', default='baseline')
    main(parser)