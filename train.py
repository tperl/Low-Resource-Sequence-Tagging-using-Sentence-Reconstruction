import argparse
import numpy as np
from keras.optimizers import Adam
from keras.models import Input
import keras
import os
import tensorflow as tf
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def get_data(lang):
    X_tgt_tr = np.load('../data/parsed/' + lang + '/X_muse.npy')
    X_char_tr = np.load('../data/parsed/' + lang + '/X_char.npy')
    y_tr = np.load('../data/parsed/' + lang + '/y.npy')
    X_tgt_dev = np.load('../data/parsed/' + lang + '/X_muse_dev.npy')
    X_char_dev = np.load('../data/parsed/' + lang + '/X_char_dev.npy')
    y_dev = np.load('../data/parsed/' + lang + '/y_dev.npy')
    embedding_matrix = np.load('../data/parsed/' + lang + '/embedding_matrix.npy')

    with open('../data/parsed/' + lang + '/idx2word.json') as f:
        idx2word = json.load(f)

    word2idx = {v: k for (k, v) in idx2word.items()}
    return X_tgt_tr, X_char_tr, y_tr, X_tgt_dev, X_char_dev, y_dev, embedding_matrix, idx2word, word2idx

def main(parser):
    args = parser.parse_args()
    print('--------------------------------------------------')
    print('arguments selected: ',args)
    print('--------------------------------------------------')

    lang = args.lang
    src_lang = args.src_lang

    # load target data
    X_tgt_tr, X_tgt_char_tr, y_tgt_tr, X_tgt_dev, X_tgt_char_dev, y_tgt_dev, tgt_embedding_matrix, tgt_idx2word, tgt_word2idx = get_data(lang)

    # concat train+dev
    if args.concatenate_dev_train:
        X_tgt_tr = np.concatenate((X_tgt_tr, X_tgt_dev), axis=0)
        X_tgt_char_tr = np.concatenate((X_tgt_char_tr, X_tgt_char_dev), axis=0)
        y_tgt_tr = np.concatenate((y_tgt_tr, y_tgt_dev), axis=0)

    # Vars for each corpus
    n_chars = np.max(X_tgt_char_tr) - 1
    max_len = X_tgt_tr.shape[-1]
    max_len_char = X_tgt_char_tr.shape[-1]
    n_tags = y_tgt_tr.shape[-1]

    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    if args.model == 'baseline':
        from models.baseline import BaselineModel
        word_in = Input(shape=(max_len,))
        char_in = Input(shape=(max_len, max_len_char,))
        model, crf = BaselineModel(word_in, char_in, tgt_embedding_matrix, n_chars, max_len_char, max_len, n_tags)
        in_data = [X_tgt_tr,
                   np.array(X_tgt_char_tr)]
        out_data = np.array(y_tgt_tr)
    elif args.model == 'transfer':
        from models.transfer_model import TransferModel
        word_in = Input(shape=(max_len,))
        char_in = Input(shape=(max_len, max_len_char,))
        lang_in = Input(shape=(max_len,))

        # load src data when transfer learnin is on
        X_src_tr, X_src_char_tr, y_src_tr, X_src_dev, X_src_char_dev, y_src_dev, src_embedding_matrix, src_idx2word, src_word2idx = get_data(src_lang)

        if args.concatenate_dev_train:
            X_src_tr = np.concatenate((X_src_tr, X_src_dev), axis=0)
            X_src_char_tr = np.concatenate((X_src_char_tr, X_src_char_dev), axis=0)
            y_src_tr = np.concatenate((y_src_tr, y_src_dev), axis=0)

        # Final dataset is both src and target, could be expanded to multiple languages
        X_tr = np.concatenate((X_tgt_tr, X_tgt_tr, X_src_tr), axis=0)
        X_char = np.concatenate((X_tgt_char_tr,X_tgt_char_tr, X_src_char_tr), axis=0)
        y_tr = np.concatenate((y_tgt_tr,y_tgt_tr, y_src_tr), axis=0)

        # language selector input
        X_lang_tr = np.full((X_tr.shape[0], X_tr.shape[1]), True, dtype=bool)
        X_lang_tr[X_tgt_tr.shape[0]:] = False

        # create keras model
        model, crf = TransferModel(word_in, char_in, lang_in, src_embedding_matrix, tgt_embedding_matrix, n_chars, max_len_char, max_len, n_tags)
        in_data = [X_tr,
                   np.array(X_char), X_lang_tr]
        out_data = np.array(y_tr)

    if args.model_weights is not None:
        model.load_weights(args.model_weights)

    optimizer = Adam(float(args.learning_rate))
    model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])

    sm_cb = keras.callbacks.ModelCheckpoint('saved_models/model_'+args.model+'_snapshot.h5', monitor='crf_viterbi_accuracy', verbose=0, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)
    rlr_cb = keras.callbacks.ReduceLROnPlateau(monitor='crf_viterbi_accuracy', factor=0.1, patience=5, verbose=0, mode='auto',
                                               min_delta=0.0001, cooldown=0, min_lr=0)
    # train model
    model.fit(in_data, out_data,
              batch_size=args.batch_size,
              epochs=args.num_of_epochs,
              verbose=args.verbosity,
              callbacks=[sm_cb, rlr_cb])

    model.save_weights('saved_models/model_'+args.model+'.h5')
    print('Done!')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='What Language to train',default='dutch')
    parser.add_argument('--src_lang', help='What Language to train', default='english')
    parser.add_argument('--concatenate_dev_train', help='Flag for concatentating train with dev', default=True)
    parser.add_argument('--model',
                        help='Flag for what model to choose: baseline/L2/transfer',
                        default='baseline')
    parser.add_argument('--model_weights',
                        help='Flag for what model to choose: baseline/L2/transfer',
                        default=None)
    parser.add_argument('--batch_size', help='batch_size', default=256)
    parser.add_argument('--num_of_epochs', help='num_of_epochs', default=50)
    parser.add_argument('--verbosity', help='verbosity for printing during training', default=1)
    parser.add_argument('--learning_rate', help='learning rate for training', default=0.01)
    main(parser)