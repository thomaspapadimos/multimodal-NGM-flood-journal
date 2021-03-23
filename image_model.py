import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from scipy.stats import entropy
import preprocessor as p
import math
import functools
import time
from nltk.corpus import stopwords
from datetime import date
from datetime import time
import nltk
import seaborn as sns

import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas
import datetime
from sklearn.model_selection import train_test_split
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import config
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge




#------------------------------------------------------


def create_img_features_from_files(df,mod):
    import requests
    from tensorflow.keras.preprocessing import image
    import numpy as np

    if mod == 'vgg19':
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'xception':
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'resnet152v2':
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'inceptionresnetv2':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'densenet201':
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling='max')
    elif mod == 'nasnetlarge':
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling='max')
    import time
    print('Loading image feature vectors...')
    start = time.time()
    images = np.zeros((len(df), model.output_shape[1]))
    n = 0
    for i in df['image_url']:
        #print(n)
        if i == 0:
            a = 0
            n = n + 1
        else:
            try:
                img_path = (config.DATA_PATH + "/images/sample_image%d.jpg" % n)
                if mod == 'nasnetlarge':
                    img = image.load_img(img_path, target_size=(331, 331))
                else:
                    img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                images[n] = features
                n = n + 1
            except FileNotFoundError:

                response = requests.get(i)
                if response.status_code == 200:
                    print('file%d not found,download again' % n)
                    file = open((config.DATA_PATH + "/images/sample_image%d.jpg" % n), "wb")
                    file.write(response.content)
                    file.close()
                    img_path = (config.DATA_PATH + "/images/sample_image%d.jpg" % n)
                    if mod == 'nasnetlarge' :
                      img = image.load_img(img_path, target_size=(331, 331))
                    else:
                      img = image.load_img(img_path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = model.predict(x)
                    images[n] = features
                n = n + 1

    print('Loaded in %f seconds' % (time.time() - start))
    save_array_to_file(images,mod)
    return images


def save_array_to_file(data, name):
    # save numpy array as csv file
    from numpy import asarray
    from numpy import savetxt
    import config
    # save to csv file
    savetxt(config.DATA_PATH + '/%s.csv'% name, data, delimiter=',')


def load_array_from_file():
    # load numpy array from csv file
    from numpy import loadtxt
    # load array
    data = loadtxt('data.csv', delimiter=',')
    # print the array
    return data
