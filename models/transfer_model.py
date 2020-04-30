from keras.models import Model, Input
from keras.layers import CuDNNLSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional,Lambda
from keras.layers import SpatialDropout1D, concatenate, Add, Conv1D
from keras_contrib.layers import CRF
import keras.backend as K

def TransferModel(word_in,
                  char_in,
                  lang_in,
                  src_embedding_matrix,
                  tgt_embedding_matrix,
                  n_chars,
                  max_len_char,
                  max_len,
                  n_tags,
                  add_reconstruction):
    h1 = 100
    h2 = 100
    h3 = 100
    output_dim = 80
    embedding_size = src_embedding_matrix.shape[1]

    def wordL2Norm(x):
        return K.l2_normalize(x, axis=2)
    def sentenceL2Norm(x):
        return K.l2_normalize(x, axis=(1,2))
    def switchLambda(x):
        return K.switch(x[0],x[1],x[2])

    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=output_dim,
                                         input_length=max_len_char, mask_zero=False))(char_in)
    # character LSTM to get word encodings by characters
    # char_enc = TimeDistributed(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.3,dropout=0.3)))(emb_char)
    char_enc = TimeDistributed(Bidirectional(CuDNNLSTM(units=h1, return_sequences=False)))(emb_char)

    embedding_src = Embedding(input_dim=src_embedding_matrix.shape[0],
                              output_dim=src_embedding_matrix.shape[1],
                              input_length=max_len,
                              weights=[src_embedding_matrix],
                              trainable=False,
                              name='src_embedding_layer')(word_in)

    embedding_tgt = Embedding(input_dim=tgt_embedding_matrix.shape[0],
                              output_dim=tgt_embedding_matrix.shape[1],
                              input_length=max_len,
                              weights=[tgt_embedding_matrix],
                              trainable=False,
                              name='tgt_embedding_layer')(word_in)

    concat_src = concatenate([char_enc, embedding_src])
    concat_tgt = concatenate([char_enc, embedding_tgt])
    embedding_switch = Lambda(switchLambda)([lang_in, concat_tgt, concat_src])

    # encoder
    encoder_tgt = Bidirectional(CuDNNLSTM(units=h2, return_sequences=True), name='encoder_tgt')(embedding_switch)  #
    # encoder_src = Bidirectional(CuDNNLSTM(units=h2, return_sequences=True), name='encoder_src')(concat_src)  #
    # encoder_switch = Lambda(switchLambda)([lang_in, encoder_tgt, encoder_src])

    fv = Lambda(sentenceL2Norm, name='feature_vector')(encoder_tgt)
    model = TimeDistributed(Dropout(0.35))(fv)
    model = Bidirectional(CuDNNLSTM(units=h3, return_sequences=True))(model)  # variational biLSTM
    model = TimeDistributed(Dropout(0.35))(model)
    crf = CRF(units=n_tags)  # CRF layer
    model = crf(model)  # outpu

    # Reconstruction branch
    if add_reconstruction:
        d1 = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')(fv)
        d2 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(d1)
        d3 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(d2)
        d4 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(d3)
        decoder_out = Conv1D(filters=embedding_size, kernel_size=1, padding='same', use_bias=False)(d4)
        decoder_out = Lambda(wordL2Norm)(decoder_out)
        model = Model([word_in, char_in, lang_in], [model, decoder_out])
    else:
        model = Model([word_in, char_in, lang_in], model)
    return model, crf