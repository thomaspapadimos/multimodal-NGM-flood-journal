import keras
from keras import optimizers, regularizers
from keras.layers import Dense, Dropout
from keras.layers import Input, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.merge import Dot
from keras.models import Model
from keras.layers import Bidirectional, LSTM
import metrics
import tensorflow as tf

#from tensorflow.keras import layers, Model, backend as K
#from layers import Time2Vec


def make_gnn_bilstm_model(HPARAMS):

  """Builds a bi-directional LSTM model."""
  inputs = Input(shape=(HPARAMS.max_seq_length,), dtype='int64', name='words')
  embedding_layer = Embedding(input_dim = HPARAMS.vocab_size,
                                              output_dim = HPARAMS.num_embedding_dims,
                                              input_length= HPARAMS.max_seq_length)(inputs)
  conv_d = Conv1D(filters=100,kernel_size=5,strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-5),name='document_conv')(embedding_layer)
  #pool_d = GlobalMaxPooling1D(name='document_pool', pool_size=2)(conv_d)
  flatten = keras.layers.MaxPooling1D(pool_size=2)(conv_d)

  lstm_layer1 = Bidirectional(LSTM(units = HPARAMS.num_lstm_dims,dropout=0.2, return_sequences=True))(flatten)
  lstm_layer = Bidirectional(LSTM(units = HPARAMS.num_lstm_dims,dropout=0.2))(lstm_layer1)
  dense_layer = Dense(HPARAMS.num_fc_units, activation='relu')(lstm_layer)
  outputs = Dense(1, activation='sigmoid')(dense_layer)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
  model.summary()
  return model

def sm_model(embed_dim,  max_doc_len, vocab_size, embeddings, addit_feat_len, img_feat_len,bert_len, no_conv_filters=100):
    """Neural architecture as mentioned in the original paper."""
    print('Preparing model with the following parameters: ')
    print('''embed_dim, max_ques_len, max_ans_len, vocab_size, embedding, addit_feat_len, no_conv_filters: ''', )
    print(embed_dim,  max_doc_len, vocab_size, embeddings.shape, addit_feat_len, no_conv_filters)

    # Prepare layers for Document
    input_d = Input(shape=(max_doc_len,), name='words')

    # Load embedding values from corpus here.
    embed_d = Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=max_doc_len,
                        weights=[embeddings],
                        trainable=False)(input_d)

    #CNN approach
    #conv_d = Conv1D(filters=no_conv_filters,kernel_size=5,strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-5),name='document_conv')(embed_d)
    #conv_d = Dropout(0.5)(conv_d)
    #pool_d = GlobalMaxPooling1D(name='document_pool')(conv_d)
    #text = Dense(100, activation='relu')(pool_d)
    #text = Dropout(0.1)(text)

    #LSTM approach
    conv_d = Bidirectional(LSTM(units=50, dropout=0.2,return_sequences=True))(embed_d)
    text = Bidirectional(LSTM(units=50, dropout=0.2))(conv_d)
    text = Dense(100, activation='relu')(text)


    # Input additional features.
    input_image_feat = Input(shape = (img_feat_len,), name='img')
    input_image_feat2 = Dense(600, activation='relu')(input_image_feat)
    input_image_feat2=Dropout(0.1)(input_image_feat2)

    input_additional_feat = Input(shape=(addit_feat_len,), name='temp')
    input_additional_feat2 = Dense(10, activation='relu')(input_additional_feat)
    input_additional_feat2 = Dropout(0.1)(input_additional_feat2)



    #input_bert_feat = Input(shape = (bert_len,), name = 'bert')
    #input_bert_feat2 = Dense(1000, activation='relu')(input_bert_feat)
    #input_bert_feat2 = Dropout(0.1)(input_bert_feat2)

    join_layer = keras.layers.concatenate([text,input_image_feat2,input_additional_feat2], name = 'concat_layer')
    #join_layer = keras.layers.concatenate([pool_d, input_additional_feat])
    #join_layer = keras.layers.concatenate([pool_d,co_image_feature,input_image_feat, input_additional_feat])


    join_layer = Dropout(0.5)(join_layer)

    hidden_units = join_layer.shape[1]

    hidden_layer = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(1e-3),name='hidden_layer')(join_layer)
    #hidden_layer = Dense(units=hidden_units, activation='relu',name='hidden_layer')(join_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    #model = Model(inputs=[input_d, input_additional_feat, input_image_feat,input_bert_feat], outputs=softmax_layer)
    model = Model(inputs=[input_d, input_image_feat, input_additional_feat], outputs=softmax_layer)

    #ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
    model.summary()

    return model


def sm_model_BERT_IMG_TEMP(embed_dim,  max_doc_len, vocab_size, embeddings, addit_feat_len, img_feat_len,bert_len, no_conv_filters=100):
    """Neural architecture as mentioned in the original paper."""
    print('Preparing model with the following parameters: ')
    print('''embed_dim, max_ques_len, max_ans_len, vocab_size, embedding, addit_feat_len, no_conv_filters: ''', )
    print(embed_dim,  max_doc_len, vocab_size, embeddings.shape, addit_feat_len, no_conv_filters)



    # Input additional features.
    input_bert_feat = Input(shape = (bert_len,), name = 'bert')
    input_bert_feat2 = Dense(1000, activation='relu')(input_bert_feat)
    input_bert_feat2 = Dropout(0.1)(input_bert_feat2)

    input_image_feat = Input(shape = (img_feat_len,), name='img')
    input_image_feat2 = Dense(600, activation='relu')(input_image_feat)
    input_image_feat2=Dropout(0.1)(input_image_feat2)

    input_additional_feat = Input(shape=(addit_feat_len,), name='temp')
    input_additional_feat2 = Dense(10, activation='relu')(input_additional_feat)
    input_additional_feat2 = Dropout(0.1)(input_additional_feat2)



    join_layer = keras.layers.concatenate([input_bert_feat2,input_image_feat2,input_additional_feat2], name = 'concat_layer')
    #join_layer = keras.layers.concatenate([pool_d, input_additional_feat])
    #join_layer = keras.layers.concatenate([pool_d,co_image_feature,input_image_feat, input_additional_feat])


    join_layer = Dropout(0.5)(join_layer)

    hidden_units = join_layer.shape[1]

    hidden_layer = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(1e-3),name='hidden_layer')(join_layer)
    #hidden_layer = Dense(units=hidden_units, activation='relu',name='hidden_layer')(join_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    #model = Model(inputs=[input_d, input_additional_feat, input_image_feat,input_bert_feat], outputs=softmax_layer)
    model = Model(inputs=[input_bert_feat, input_image_feat, input_additional_feat], outputs=softmax_layer)

    #ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
    model.summary()

    return model

def sm_model_TEXT_IMG(embed_dim,  max_doc_len, vocab_size, embeddings, addit_feat_len, img_feat_len,bert_len, no_conv_filters=100):
    """Neural architecture as mentioned in the original paper."""
    print('Preparing model with the following parameters: ')
    print('''embed_dim, max_ques_len, max_ans_len, vocab_size, embedding, addit_feat_len, no_conv_filters: ''', )
    print(embed_dim,  max_doc_len, vocab_size, embeddings.shape, addit_feat_len, no_conv_filters)
    # Prepare layers for Document
    input_d = Input(shape=(max_doc_len,), name='words')
    # Load embedding values from corpus here.
    embed_d = Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=max_doc_len,
                        weights=[embeddings],
                        trainable=False)(input_d)

    #CNN approach
    #conv_d = Conv1D(filters=no_conv_filters,kernel_size=5,strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-5),name='document_conv')(embed_d)
    #conv_d = Dropout(0.5)(conv_d)
    #pool_d = GlobalMaxPooling1D(name='document_pool')(conv_d)
    #text = Dense(100, activation='relu')(pool_d)
    #text = Dropout(0.1)(text)

    #LSTM approach
    conv_d = Bidirectional(LSTM(units=50, dropout=0.2,return_sequences=True))(embed_d)
    text = Bidirectional(LSTM(units=50, dropout=0.2))(conv_d)
    text = Dense(100, activation='relu')(text)


    # Input additional features.
    input_image_feat = Input(shape = (img_feat_len,), name='img')
    input_image_feat2 = Dense(600, activation='relu')(input_image_feat)
    input_image_feat2=Dropout(0.1)(input_image_feat2)


    join_layer = keras.layers.concatenate([text,input_image_feat2], name = 'concat_layer')
    #join_layer = keras.layers.concatenate([pool_d, input_additional_feat])
    #join_layer = keras.layers.concatenate([pool_d,co_image_feature,input_image_feat, input_additional_feat])
    join_layer = Dropout(0.5)(join_layer)

    hidden_units = join_layer.shape[1]

    hidden_layer = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(1e-3),name='hidden_layer')(join_layer)
    #hidden_layer = Dense(units=hidden_units, activation='relu',name='hidden_layer')(join_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    #model = Model(inputs=[input_d, input_additional_feat, input_image_feat,input_bert_feat], outputs=softmax_layer)
    model = Model(inputs=[input_d, input_image_feat], outputs=softmax_layer)

    #ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
    model.summary()

    return model




def sm_model_TEXT_TEMP(embed_dim,  max_doc_len, vocab_size, embeddings, addit_feat_len, img_feat_len,bert_len, no_conv_filters=100):
    """Neural architecture as mentioned in the original paper."""
    print('Preparing model with the following parameters: ')
    print('''embed_dim, max_ques_len, max_ans_len, vocab_size, embedding, addit_feat_len, no_conv_filters: ''', )
    print(embed_dim,  max_doc_len, vocab_size, embeddings.shape, addit_feat_len, no_conv_filters)
    # Prepare layers for Document
    input_d = Input(shape=(max_doc_len,), name='words')
    # Load embedding values from corpus here.
    embed_d = Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=max_doc_len,
                        weights=[embeddings],
                        trainable=False)(input_d)

    #CNN approach
    #conv_d = Conv1D(filters=no_conv_filters,kernel_size=5,strides=1, padding='same', activation='tanh', kernel_regularizer=regularizers.l2(1e-5),name='document_conv')(embed_d)
    #conv_d = Dropout(0.5)(conv_d)
    #pool_d = GlobalMaxPooling1D(name='document_pool')(conv_d)
    #text = Dense(100, activation='relu')(pool_d)
    #text = Dropout(0.1)(text)

    #LSTM approach
    conv_d = Bidirectional(LSTM(units=50, dropout=0.2,return_sequences=True))(embed_d)
    text = Bidirectional(LSTM(units=50, dropout=0.2))(conv_d)
    text = Dense(100, activation='relu')(text)


    # Input additional features.

    input_additional_feat = Input(shape=(addit_feat_len,), name='temp')
    input_additional_feat2 = Dense(10, activation='relu', name='temp_dense')(input_additional_feat)
    input_additional_feat2 = Dropout(0.1)(input_additional_feat2)

    join_layer = keras.layers.concatenate([text,input_additional_feat2], name = 'concat_layer')
    #join_layer = keras.layers.concatenate([pool_d, input_additional_feat])
    #join_layer = keras.layers.concatenate([pool_d,co_image_feature,input_image_feat, input_additional_feat])
    join_layer = Dropout(0.5)(join_layer)

    hidden_units = join_layer.shape[1]

    hidden_layer = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(1e-3),name='hidden_layer')(join_layer)
    #hidden_layer = Dense(units=hidden_units, activation='relu',name='hidden_layer')(join_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    #model = Model(inputs=[input_d, input_additional_feat, input_image_feat,input_bert_feat], outputs=softmax_layer)
    model = Model(inputs=[input_d, input_additional_feat], outputs=softmax_layer)

    #ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
    model.summary()

    return model


def softmax_model(input_dim):
    """Neural architecture as mentioned in the original paper."""

    # Prepare layers for Document
    input = Input(shape=(input_dim,), name='words')

    hidden_layer = Dense(units=input.shape[1],activation='relu',kernel_regularizer=regularizers.l2(1e-5),name='hidden_layer')(input)
    #hidden_layer = Dense(units=hidden_units, activation='relu',name='hidden_layer')(input)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    #model = Model(inputs=[input_d, input_additional_feat, input_image_feat,input_bert_feat], outputs=softmax_layer)
    model = Model(inputs=[input], outputs=softmax_layer)

    #ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.precision, metrics.recall, metrics.f1])
    model.summary()

    return model