"""

Sentiment analysis usong LSTM from:
    - https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb

Introduction:

Sentiment analysis can be thought of as the exercise of taking a sentence, paragraph, document, or any piece of
natural language, and determining whether that text's emotional tone is positive, negative or neutral.

Topics:
- Word vectors,
- recurrent neural networks, and
- long short-term memory units (LSTMs)
"""

# GloVe embedding
VOCAB_SIZE = 400000 # total number of words in vocabulary
VOCAB_EMB_DIM = 50  # embedding size for each word

NUM_POSITIVE_FILES = 12500
NUM_NEGATIVE_FILES = 12500

WORDS_LIST = 'wordsList.npy'
WORD_VECTORS = 'wordsList.npy'

POSTIVE_PATH = 'positiveReviews/'
NEGATIVE_PATH = 'negativeReviews/'
CP_PATH = "models/pretrained_lstm.ckpt"

MAX_SEQ_LENGTH = 250    # maximum length of one review sentence
UNK_ID = 399999
BATCH_SIZE = 24
LSTM_UNITS = 64
ITERATIONS = 100000
LEARNING_RATE = .001


