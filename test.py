import argparse
import numpy as np
from keras.optimizers import Adam
from keras.models import Input
from utils.f1_test import f1_test
import json
import os


def run_test(X_muse_test, X_char_test, idx2tag, model, max_len, y_test, debug_mode=False, X_lang=None):
    pred_tags = []
    for i in range(0,int(len(X_muse_test)/100)):
        if X_lang is None:
            preds = model.predict_on_batch([X_muse_test[i*100:(i+1)*100], np.array(X_char_test[i*100:(i+1)*100])])
        else:
            preds = model.predict_on_batch(
                [X_muse_test[i * 100:(i + 1) * 100], np.array(X_char_test[i * 100:(i + 1) * 100]), X_lang[i * 100:(i + 1) * 100]])
        if type(preds) == list:
            preds = preds[0]
        preds = np.argmax(preds, axis=2)

        for p in preds:
            pred_tags.append([idx2tag[str(p[j])] for j in range(p.shape[0])])

    i+=1
    i = i*100
    while i <len(X_muse_test):
        if X_lang is None:
            preds = model.predict_on_batch([X_muse_test[i * 1:(i + 1) * 1], np.array(X_char_test[i * 1:(i + 1) * 1])])
        else:
            preds = model.predict_on_batch([X_muse_test[i * 1:(i + 1) * 1], np.array(X_char_test[i * 1:(i + 1) * 1]), X_lang[i * 1:(i + 1) * 1]])
        if type(preds) == list:
            preds = preds[0]
        preds = np.argmax(preds,axis=2)[0]

        pred_tags.append([idx2tag[str(preds[j])] for j in range(len(preds))])
        i+= 1
    gt_tags = np.argmax(y_test,axis=2)
    gt_tags = [[idx2tag[str(gt_tags[j][i])] for i in range(max_len)] for j in range(X_muse_test.shape[0])]
    return f1_test(gt_tags,pred_tags, debug_mode)


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
        model, crf = BaselineModel(word_in, char_in, embedding_matrix, n_chars, max_len_char, max_len, n_tags, args.add_reconstruction)

        model.load_weights('saved_models/model_baseline.h5')
        f1, precision, recall = run_test(X_tgt_muse_test, X_char_test, idx2tag, model, max_len, y_test)
    elif args.model == 'transfer':
        from models.transfer_model import TransferModel
        word_in = Input(shape=(max_len,))
        char_in = Input(shape=(max_len, max_len_char,))
        lang_in = Input(shape=(max_len,))
        model, crf = TransferModel(word_in, char_in, lang_in, embedding_matrix, embedding_matrix, n_chars, max_len_char, max_len, n_tags, args.add_reconstruction)
        X_lang = np.full((X_tgt_muse_test.shape[0], X_tgt_muse_test.shape[1]), True, dtype=bool)
        model.load_weights('saved_models/model_transfer.h5')
        f1, precision, recall = run_test(X_tgt_muse_test, X_char_test, idx2tag, model, max_len, y_test,debug_mode=True, X_lang=X_lang)

    if args.log_results is not None:
        with open(args.log_results,'w') as f_out:
            out_string = 'f1='+str(f1) + '\nprecision='+ str(precision) + '\nrecall='+str(recall)
            f_out.writelines(out_string)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='What Language to train',default='dutch')
    parser.add_argument('--src_lang', help='What Language to train', default='english')
    parser.add_argument('--add_reconstruction', help='Flag for conc`atentating train with dev', default=False)
    parser.add_argument('--model', help='Flag for what model to choose baseline/L2/', default='transfer')
    parser.add_argument('--log_results', help='location to log test results/', default=None)
    main(parser)