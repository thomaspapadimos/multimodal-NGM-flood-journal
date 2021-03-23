import neural_structured_learning as nsl

DATA_PATH = 'data/MediaEval'
DATASET_PATH = DATA_PATH + '/dataset.json'
EMBEDDING_PATH = DATA_PATH + '/itwiki100d.txt'
IMAGE_MODALITY = 'vgg19'
VOCAB_PATH = DATA_PATH + 'vocab.json'

QUERY_MAX_SEQUENCE_LENGTH = 8
DOC_MAX_SEQUENCE_LENGTH = 50

SPLITS = 5
VALIDATION_SPLIT = 0.1
EPOCHS = 35
RANDOM_STATE = 42
BATCH_SIZE = 64

### neural graph learning parameters
distance_type = nsl.configs.DistanceType.L2
graph_regularization_multiplier = 0.1
num_neighbors = 2
similarity_threshold = 0.5