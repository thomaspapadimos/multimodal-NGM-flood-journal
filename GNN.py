
import config
import models2
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import pickle
import json
import numpy as np
from numpy import loadtxt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import metrics
from text_preprocessing import create_vocab, load_embeddings_from_file, get_sequences, load_embeddings, save_embeddings
from batch_generator import batch_gen
from sklearn.metrics import average_precision_score
from simpletransformers.language_representation import RepresentationModel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
import image_model
import pandas as pd
import gensim
import text_preprocessing
import bert_model
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import neural_structured_learning as nsl

import tensorflow as tf
import tensorflow_hub as hub


tf.keras.backend.clear_session()

NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'




def make_dataset(file_path,HPARAMS, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """

  def pad_sequence(sequence, max_seq_length):
    """Pads the input sequence (a `tf.SparseTensor`) to `max_seq_length`."""
    pad_size = tf.maximum([0], max_seq_length - tf.shape(sequence)[0])
    padded = tf.concat(
        [sequence.values,
         tf.fill((pad_size), tf.cast(0, sequence.dtype))],
        axis=0)
    # The input sequence may be larger than max_seq_length. Truncate down if
    # necessary.
    return tf.slice(padded, [0], [max_seq_length])

  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    # The 'words' feature is a variable length word ID vector.
    feature_spec = {
        'words': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }
    # We also extract corresponding neighbor features in a similar manner to
    # the features above during training.
    if training:
      for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.VarLenFeature(tf.int64)

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)

    # Since the 'words' feature is a variable length word vector, we pad it to a
    # constant maximum length based on HPARAMS.max_seq_length
    features['words'] = pad_sequence(features['words'], HPARAMS.max_seq_length)
    if training:
      for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        features[nbr_feature_key] = pad_sequence(features[nbr_feature_key],
                                                 HPARAMS.max_seq_length)

    labels = features.pop('label')
    return features, labels
  dataset = tf.data.TFRecordDataset([file_path])
  if training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(HPARAMS.batch_size)
  return dataset


def create_all_data(dir,df_tweet,img_mod):
    with open(config.DATA_PATH + '/b.seq') as f:
        test_document_sequences = [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]
        test_document_sequences = pad_sequences(test_document_sequences, maxlen=config.DOC_MAX_SEQUENCE_LENGTH)

    text = np.array(test_document_sequences)

    #df_tweet = pd.read_json('aqua3S.json')
    df_tweet['relevant_int'] = df_tweet['relevant'].astype(int)
    labels = np.array(df_tweet['relevant_int'].values)

    df_tweet['temp_feature'] = text_preprocessing.temp_feature(df_tweet)
    temp_feat = np.array(df_tweet['temp_feature'].values)

    df_tweet['image_url'] = df_tweet['image_url'].fillna(0)
    #img_feat = image_model.create_img_features_from_files(df_tweet, 'vgg19')
    #img_feat = image_model.download_create_img_features(df_tweet,'vgg19')
    try:
        img_feat = loadtxt(config.DATA_PATH +'/%s.csv'%img_mod, delimiter=',')
    except OSError:
        img_feat = image_model.create_img_features_from_files(df_tweet, img_mod) #downloads the nonexist images
        #img_feat = image_model.download_create_img_features(df_tweet,img_mod)
    try:
        bert_feat = loadtxt(config.DATA_PATH +'/setence_repr_bert.csv', delimiter = ',')
    except:
        bert_feat = bert_model.setence_representation(df_tweet,config.DATA_PATH)
        #bert_feat = bert_model.word_representation(df_tweet,config.DATA_PATH)

    return text, labels, temp_feat, img_feat, bert_feat

def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


def create_embedding_example(word_vector, record_id):
  """Create tf.Example containing the sample's embedding and its ID."""

  embedding = tf.reshape(word_vector, shape=[-1])

  features = {
      'id': _bytes_feature(str(record_id)),
      'embedding': _float_feature(embedding.numpy())
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def create_embeddings(word_vectors, output_path, starting_record_id):
  record_id = int(starting_record_id)
  with tf.io.TFRecordWriter(output_path) as writer:
    for word_vector in word_vectors:
      example = create_embedding_example(word_vector, record_id)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id

def create_example(word_vector, label, record_id):
  """Create tf.Example containing the sample's word vector, label, and ID."""
  features = {
      'id': _bytes_feature(str(record_id)),
      'words': _int64_feature(np.asarray(word_vector)),
      'label': _int64_feature(np.asarray([label])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))

def create_records(word_vectors, labels, record_path, starting_record_id):
  record_id = int(starting_record_id)
  with tf.io.TFRecordWriter(record_path) as writer:
    for word_vector, label in zip(word_vectors, labels):
      example = create_example(word_vector, label, record_id)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id

def train_model():
    import neural_structured_learning as nsl
    import tensorflow as tf
    import tensorflow_hub as hub

    # Resets notebook state
    tf.keras.backend.clear_session()

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print(
        "GPU is",
        "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


    import nltk
    print('clean italian tweet text')

    if config.DATA_PATH == 'data/MediaEval':
        df_tweet = pd.read_json(config.DATA_PATH+ '/beaware.json')
        df_mediaeval_dev = pd.read_json(config.DATA_PATH + '/devset_tweets_ids.json')
        df_mediaeval_test = pd.read_json(config.DATA_PATH + '/testset_tweets_ids.json')
        df_mediaeval = pd.concat([df_mediaeval_dev, df_mediaeval_test])
        df_mediaeval = df_mediaeval.rename(columns={0: 'id_str'})
        df = df_mediaeval.merge(df_tweet, how='inner', indicator=False)
        df_tweet = df
    else:
        df_tweet = pd.read_json(config.DATASET_PATH)
        df_tweet = df_tweet.sort_values(by=['timestamp_ms'], ascending=False)

    nltk.download('punkt')
    df_tweet['tweet_clean'] = df_tweet['full_text'].apply(text_preprocessing.tweet_preprocess)
    df_tweet['clean'] = df_tweet['tweet_clean'].apply(text_preprocessing.preprocess)
    df_tweet['clean_joined'] = df_tweet['clean'].apply(lambda x: " ".join(x))
    df_tweet['clean_joined'].replace('', np.nan, inplace=True)
    df_tweet.dropna(subset=['clean_joined'], inplace=True)
    df_tweet = df_tweet.reset_index()
    maxlen = -1
    for doc in df_tweet.clean_joined:
        tokens = nltk.word_tokenize(doc)
        if (maxlen < len(tokens)):
            maxlen = len(tokens)
    print("The maximum number of words in any document is =", maxlen)

    f = open(config.DATA_PATH + "/b.toks", "w+")
    for j in range(len(df_tweet)):
        f.writelines(str(df_tweet.clean_joined[j]) + '\n')
    f.close()

    texts = []
    with open(config.DATA_PATH + '/b.toks') as f:
        texts.extend([line for line in f])
    vocab, tokenizer = create_vocab(texts)


    with open(config.DATA_PATH + '/b.toks') as f:
        b_seq = get_sequences(tokenizer, [line for line in f])
        with open(config.DATA_PATH + '/b.seq', 'w+', newline='') as o_f:
            wr = csv.writer(o_f)
            wr.writerows(b_seq)

    try:
        embeddings, embed_dim, _ = load_embeddings_from_file(config.DATA_PATH)
        print('embedding dim:', embed_dim)
    except:
        embeddings, embed_dim, _ = load_embeddings(config.EMBEDDING_PATH, vocab)
        print('embedding dim:', embed_dim)
        save_embeddings(embeddings, config.DATA_PATH)

    text, labels, temp_feat, img_feat, bert = create_all_data(config.DATA_PATH, df_tweet, config.IMAGE_MODALITY)


    clear_session()
    for split in range(config.SPLITS):

        keras.backend.clear_session()

        if config.DATA_PATH == 'data/MediaEval':
            pp_train_data1 = text[0:5418]
            pp_test_data = text[5418:]

            pp_train_labels1 = labels[0:5418]
            pp_test_labels = labels[5418:]

            bert_train1 = bert[0:5418]
            bert_test = bert[5418:]

            img_train1 = img_feat[0:5418]
            img_test = img_feat[5418:]

            ros = RandomUnderSampler(sampling_strategy=0.7,random_state = 42)
            ros.fit(pp_train_data1, pp_train_labels1)
            pp_train_data, pp_train_labels = ros.fit_sample(pp_train_data1, pp_train_labels1)
            bert_train, pp_train_labels = ros.fit_sample(bert_train1, pp_train_labels1)
            img_train, pp_train_labels = ros.fit_sample(img_train1, pp_train_labels1)
        else:
            pp_train_data, pp_test_data, pp_train_labels, pp_test_labels, bert_train, bert_test, img_train, img_test, temp_train, temp_test = train_test_split(
                text, labels, bert,img_feat,temp_feat, test_size=0.2, shuffle=True)
            addit_feat_len = temp_train.shape[1] if temp_train.ndim > 1 else 1
            img_feat_len = img_train.shape[1] if img_train.ndim > 1 else 1
            bert_len = bert_train.shape[1] if bert_train.ndim > 1 else 1



        print('Training entries: {}, labels: {}'.format(len(pp_train_data), len(pp_train_labels)))
        training_samples_count = len(pp_train_data)

        create_embeddings(bert_train, config.DATA_PATH + '/embeddings.tfr', 0)



        graph_builder_config = nsl.configs.GraphBuilderConfig(similarity_threshold=0.5, lsh_splits=32, lsh_rounds=15, random_seed=12345)
        nsl.tools.build_graph_from_config([config.DATA_PATH + '/embeddings.tfr'],
                                          config.DATA_PATH + '/graph_99.tsv',
                                          graph_builder_config)
        # Persist TF.Example features (word vectors and labels) for training and test
        # data in TFRecord format.
        next_record_id = create_records(pp_train_data, pp_train_labels,
                                        config.DATA_PATH + '/train_data.tfr', 0)
        create_records(pp_test_data, pp_test_labels, config.DATA_PATH + '/test_data.tfr',
                       next_record_id)

        nsl.tools.pack_nbrs(
            config.DATA_PATH + '/train_data.tfr',
            '',
            config.DATA_PATH + '/graph_99.tsv',
            config.DATA_PATH + '/nsl_train_data.tfr',
            add_undirected_edges=True,
            max_nbrs=3)

        class HParams(object):
            """Hyperparameters used for training."""

            def __init__(self, vocab_size, epochs):
                ### dataset parameters
                self.num_classes = 2
                self.max_seq_length = 50
                self.vocab_size = vocab_size
                ### neural graph learning parameters
                self.distance_type = nsl.configs.DistanceType.L2
                self.graph_regularization_multiplier = 0.1
                self.num_neighbors = 2
                ### model architecture
                self.num_embedding_dims = 100
                self.num_lstm_dims = 64
                self.num_fc_units = 64
                ### training parameters
                self.train_epochs = epochs
                self.batch_size = 64
                ### eval parameters
                self.eval_steps = None  # All instances in the test set are evaluated.

        HPARAMS = HParams(vocab_size=len(vocab) + 1, epochs=config.EPOCHS)
        train_dataset = make_dataset(config.DATA_PATH + '/nsl_train_data.tfr',HPARAMS, True)
        #train_dataset = make_dataset('/tmp/imdb/nsl_train_data.tfr', False)
        test_dataset = make_dataset(config.DATA_PATH + '/test_data.tfr',HPARAMS)

        model = models2.make_gnn_bilstm_model(HPARAMS)

        validation_fraction = config.VALIDATION_SPLIT
        validation_size = int(validation_fraction *
                              int(training_samples_count / HPARAMS.batch_size))
        print(validation_size)
        validation_dataset = train_dataset.take(validation_size)
        train_dataset = train_dataset.skip(validation_size)

        callback = EarlyStopping(monitor='val_loss', patience=3, mode="min")
        print('Running splits: %d' % (split + 1))
        history = model.fit(train_dataset,validation_data=validation_dataset,epochs=5,verbose=1,callbacks=[callback])

        with open('data/splits%d_TRAINHistory.pkl'%(split+1), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        train_dict = pd.read_pickle('data/splits%d_TRAINHistory.pkl'%(split+1))
        train_dict = [i[-1] for i in train_dict.values()]
        try:
            geeky_file = open('data/splits%d_TRAINHistory.txt'%(split+1), 'wt')
            geeky_file.write(str(train_dict))
            geeky_file.close()

        except:
            print("Unable to write train to file")

        results = model.evaluate(test_dataset, steps=HPARAMS.eval_steps)
        print(results)

        y_pred = model.predict([pp_test_data])
        prediction = []
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        accuracy = accuracy_score(pp_test_labels, prediction)
        recall = recall_score(pp_test_labels, prediction)
        precision = precision_score(pp_test_labels, prediction)
        fscore = f1_score(pp_test_labels, prediction)
        aps = average_precision_score(pp_test_labels, y_pred)
        # test_acc = metrics.AUC(y_test, y_pred)
        print("Model METRICS: aver_pre: %f Accuracy : %f,  precision: %f,recall: %f , f1score: %f" % (
            aps, accuracy, precision, recall, fscore))


        test_dict = {'loss': 52, 'accuracy': accuracy, 'precision': precision,'recall':recall,'f1score': fscore}
        with open('data/splits%d_TESTHistory.pkl' % (split+1), 'wb') as file_pi:
            pickle.dump(test_dict, file_pi)

        test_dict = pd.read_pickle('data/splits%d_TESTHistory.pkl'%(split+1))
        test_dict = [i for i in test_dict.values()]
        try:
            geeky_file = open('data/splits%d_TESTHistory.txt'%(split+1), 'wt')
            geeky_file.write(str(test_dict))
            geeky_file.close()

        except:
            print("Unable to write test to file")

#---------------GNN MODEL-----------------------------------

        base_reg_model = models2.make_gnn_bilstm_model(HPARAMS)
        # Wrap the base model with graph regularization.
        graph_reg_config = nsl.configs.make_graph_reg_config(
            max_neighbors=HPARAMS.num_neighbors,
            multiplier=HPARAMS.graph_regularization_multiplier,
            distance_type=HPARAMS.distance_type,
            sum_over_axis=-1)
        graph_reg_model = nsl.keras.GraphRegularization(base_reg_model,
                                                        graph_reg_config)
        graph_reg_model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.precision,metrics.recall,metrics.f1])
        print('Running GNN splits: %d' % (split + 1))
        graph_reg_history = graph_reg_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=5,
            verbose=1,
            callbacks=[callback])


        with open('data/splits%d_GNN_TRAINHistory.pkl'%(split+1), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        train_dict = pd.read_pickle('data/splits%d_GNN_TRAINHistory.pkl'%(split+1))
        train_dict = [i[-1] for i in train_dict.values()]
        try:
            geeky_file = open('data/splits%d_GNN_TRAINHistory.txt'%(split+1), 'wt')
            geeky_file.write(str(train_dict))
            geeky_file.close()

        except:
            print("Unable to write train to file")

        graph_reg_results = graph_reg_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)
        print(graph_reg_results)

        y_pred = graph_reg_model.predict([pp_test_data])
        prediction = []
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        accuracy = accuracy_score(pp_test_labels, prediction)
        recall = recall_score(pp_test_labels, prediction)
        precision = precision_score(pp_test_labels, prediction)
        fscore = f1_score(pp_test_labels, prediction)
        aps = average_precision_score(pp_test_labels, y_pred)
        # test_acc = metrics.AUC(y_test, y_pred)
        print("Model METRICS: aver_pre: %f Accuracy : %f,  precision: %f,recall: %f , f1score: %f" % (
            aps, accuracy, precision, recall, fscore))

        test_dict = {'loss': 52, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1score': fscore}
        with open('data/splits%d_GNN_TESTHistory.pkl' % (split + 1), 'wb') as file_pi:
            pickle.dump(test_dict, file_pi)

        test_dict = pd.read_pickle('data/splits%d_GNN_TESTHistory.pkl' % (split + 1))
        test_dict = [i for i in test_dict.values()]
        try:
            geeky_file = open('data/splits%d_GNN_TESTHistory.txt' % (split + 1), 'wt')
            geeky_file.write(str(test_dict))
            geeky_file.close()

        except:
            print("Unable to write test to file")


if __name__ == '__main__':
    train_model()