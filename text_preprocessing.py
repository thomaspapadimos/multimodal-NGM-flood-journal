import time
import os
import datetime
import preprocessor as p
import numpy as np
from nltk.corpus import stopwords
from sklearn.neighbors import KernelDensity
import gensim
import nltk
import preprocessor as p
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
import statistics

# set which entities we want to replace
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.NUMBER, p.OPT.SMILEY)

tt = TweetTokenizer()

def temp_feature(df_tweet):
    import statistics
    time_sub = [(datetime.datetime.today() - df_tweet.timestamp_ms[i]) for i in range(len(df_tweet))]
    hour = list(map((lambda  x : x.total_seconds()/ 3600), time_sub))
    v = []
    for x in hour:
        v.append(x)
    h = (4 * (statistics.stdev(v) ** 5) / 3 * len(hour)) ** (-1 / 5)
    kde = KernelDensity(kernel='gaussian', bandwidth=h)
    # kde = KernelDensity(kernel='gaussian')
    hour = (np.array(hour))[:, np.newaxis]
    kde.fit(hour, y=None)
    log_dens = kde.score_samples(hour)
    kde_eval = np.exp(log_dens)

    return kde_eval


	
def preprocess(text):
    #nltk.download('stopwords')

    stop_words = stopwords.words('italian')
    stop_words.extend(['http', 'https'])
    # Obtain additional stopwords from nltk
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)

    return result

def tweet_preprocess(text):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)
    result=p.clean(text)
    return result

def normalize_text(text):
    text = p.tokenize(text)

    # replace $....$ with <...> to prevent tokenization of entities
    text = text.replace('$URL$', '<URL>')
    text = text.replace('$EMOJI$', '<EMOJI>')
    text = text.replace('$NUMBER$', '<NUMBER>')
    text = text.replace('$SMILEY$', '<SMILEY>')

    return text



def load_embeddings(embeddings_file, vocab):
    """
    Load pre-learnt word embeddings.
    Return: embedding: embedding matrix with dim |vocab| x dim
            dim: dimension of the embeddings
            rand_count: number of words not in trained embedding
    """

    print('Loading word vectors...')
    start = time.time()

    word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    print('Loaded in %f seconds' % (time.time() - start))

    dim = word_vectors['common'].shape[0]

    # Initialize an embedding of |vocab| x dim
    # word -> embedding
    embeddings = np.zeros((len(vocab) + 1, dim))

    # Count of words not having representations in our embedding file
    rand_count = 0
    for key, value in vocab.items():
        # Map word idx to its embedding vector.:
        try:
            embeddings[value] = word_vectors[key]
        except:
            # Take random values
            rand_vec = np.random.uniform(-0.25, 0.25, dim)
            embeddings[value] = rand_vec
            rand_count += 1

    print('Total time for loading embedding: %f seconds' % (time.time() - start))
    print('Number of words not in trained embedding: %d' % (rand_count))

    return embeddings, dim, rand_count


def load_embeddings_from_file(out_dir):
    try:
        print('Trying to load from npy dump.')
        embeddings = np.load(os.path.join(out_dir, 'embeddings.npy'))
        dim = embeddings.shape[1]

        embeddings[0] = np.random.uniform(-0.25, 0.25, dim)
        #embeddings[0] = np.zeros((dim, ))

        return embeddings, dim, 'NA'
    except:
        print('Load from dump failed, reading from binary.')


def save_embeddings(embeddings, out_dir):
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings)


def create_vocab(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab = tokenizer.word_index
    print('Found %s unique tokens.' % len(vocab))

    return vocab, tokenizer


def get_sequences(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return sequences


