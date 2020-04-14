from keras.models import Model, Input
from keras.layers import CuDNNLSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.layers import SpatialDropout1D, concatenate, Add
from keras_contrib.layers import CRF

def BaselineModel(word_in,
                  char_in,
                  embedding_matrix,
                  n_chars,
                  word_emb_output_dim,
                  max_len_char,
                  max_len,
                  h1,h2,h3,
                  n_tags):
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=word_emb_output_dim,
                                         input_length=max_len_char, mask_zero=False))(char_in)
    # character LSTM to get word encodings by characters
    # char_enc = TimeDistributed(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.3,dropout=0.3)))(emb_char)
    char_enc = TimeDistributed(Bidirectional(CuDNNLSTM(units=h1, return_sequences=False)))(emb_char)
    model = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      input_length=max_len,
                      weights=[embedding_matrix],
                      trainable=False,
                      name='tgt_embedding_layer')(word_in)
    model = concatenate([char_enc, model])
    model = model
    model = TimeDistributed(Dropout(0.05))(model)
    model = Bidirectional(CuDNNLSTM(units=h2, return_sequences=True))(model)  # variational biLSTM
    model = TimeDistributed(Dropout(0.5))(model)
    model = Bidirectional(CuDNNLSTM(units=h3, return_sequences=True))(model)  # variational biLSTM
    model = TimeDistributed(Dropout(0.5))(model)
    crf = CRF(units=n_tags)  # CRF layer
    model = crf(model)  # outpu

    model = Model([word_in, char_in], model)
    return model, crf