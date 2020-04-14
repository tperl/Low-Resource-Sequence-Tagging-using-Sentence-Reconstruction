import argparse
import numpy as np
from keras.optimizers import Adam
from keras.models import Input
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def main(parser):
    args = parser.parse_args()
    lang = args.lang
    X_tgt_muse_tr = np.load('../data/parsed/' + lang + '/X_muse.npy')
    X_char_tr = np.load('../data/parsed/' + lang + '/X_char.npy')
    y_tr = np.load('../data/parsed/' + lang + '/y.npy')
    X_tgt_muse_dev = np.load('../data/parsed/' + lang + '/X_muse_dev.npy')
    X_char_dev = np.load('../data/parsed/' + lang + '/X_char_dev.npy')
    y_dev = np.load('../data/parsed/' + lang + '/y_dev.npy')
    embedding_matrix = np.load('../data/parsed/' + lang + '/embedding_matrix.npy')

    if args.concatenate_dev_train:
        X_tgt_muse_tr = np.concatenate((X_tgt_muse_tr, X_tgt_muse_dev), axis=0)
        X_char_tr = np.concatenate((X_char_tr, X_char_dev), axis=0)
        y_tr = np.concatenate((y_tr, y_dev), axis=0)

    # Vars for each corpus
    n_chars = np.max(X_char_tr) - 1
    max_len = X_tgt_muse_tr.shape[-1]
    max_len_char = X_char_tr.shape[-1]
    n_tags = y_tr.shape[-1]

    if args.model == 'baseline':
        from models.baseline import BaselineModel
        word_in = Input(shape=(max_len,))
        char_in = Input(shape=(max_len, max_len_char,))
        h1 = 80
        h2 = 80
        h3 = 120
        output_dim = 50
        model, crf = BaselineModel(word_in, char_in, embedding_matrix, n_chars, output_dim, max_len_char, max_len, h1, h2, h3, n_tags)

    def closs(y_true, y_pred):
        X = crf.input
        mask = crf.input_mask
        nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
        return nloglik

    optimizer = Adam(args.learning_rate)
    model.compile(optimizer=optimizer, loss=closs, metrics=[crf.accuracy])
    model.fit([X_tgt_muse_tr,
               np.array(X_char_tr)],
              np.array(y_tr),
              batch_size=args.batch_size,
              epochs=args.num_of_epochs,
              verbose=args.verbosity)

    if args.model == 'baseline':
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        model.save_weights('saved_models/model_baseline.h5')

    print('Done!')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='What Language to train',default='english')
    parser.add_argument('--concatenate_dev_train', help='Flag for concatentating train with dev', default=True)
    parser.add_argument('--model', help='Flag for what model to choose baseline/reconstruction/transferlearning', default='baseline')
    parser.add_argument('--batch_size', help='batch_size', default=64)
    parser.add_argument('--num_of_epochs', help='batch_size', default=50)
    parser.add_argument('--verbosity', help='verbosity for printing during training', default=2)
    parser.add_argument('--learning_rate', help='learning rate for training', default=0.001)
    main(parser)