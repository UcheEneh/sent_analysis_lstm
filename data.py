# Importing the GloVe embeddeing matrix and vocabulary

import numpy as np
import tensorflow as tf

import util
import os


words_List = np.load(util.WORDS_LIST)
print('Loaded the GloVe word list')
words_List = words_List.tolist()  # originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in words_List]    # encode words as UTF-8 (necessary?)
wordVectors = np.load(util.WORD_VECTORS)
print('Loaded the GloVe word vectors!')

# to make sure everything has been loaded in correctly, we can look at the dimensions of the vocabulary list and
# the embedding matrix.
assert len(wordsList) == util.VOCAB_SIZE
assert wordVectors.shape == (util.VOCAB_SIZE, util.VOCAB_EMB_DIM)

# We can also search our word list for a word like "baseball", and then access its corresponding vector through
# the embedding matrix.
baseballIndex = wordsList.index('baseball')
print(wordVectors[baseballIndex])


"""
Our next step is taking an input sentence and then constructing its vector representation. 
Let's say that we have the input sentence "I thought the movie was incredible and inspiring".  In order to get the word 
vectors, we can use Tensorflow's embedding lookup function. This function takes in two arguments, one for the embedding 
matrix (the wordVectors matrix in our case), and one for the ids of each of the words. The ids vector can be thought of 
as the integerized representation of the training set. This is basically just the row index of each of the words. 
Let's look at a quick example to make this concrete.
"""

maxSeqLength = 10   # Max length of a sentence
numDimensions = 300 # Dimensions for each word vector

firstSentence = np.zeros((maxSeqLength), dtype='int32')
sentence = ['i', 'thought', 'the', 'movie', 'was', 'incredible', 'and', 'inspiring']
for count, s in enumerate(sentence):
    firstSentence[count] = wordsList.index(s)
# firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence) # Shows the row index for each word

# Now using tf.nn.embedding_lookup, we can find the embedding vector of the input sentence
# The 10 x 50 output should contain the 50 dimensional word vectors for each of the 10 words in the sequence.
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors, firstSentence).eval().shape)


# Determine total and average number of words in each review.
positiveFiles = [util.POSTIVE_PATH + f for f in os.listdir(util.POSTIVE_PATH)
                 if os.path.isfile(os.path.join(util.POSTIVE_PATH, f))]
negativeFiles = [util.NEGATIVE_PATH + f for f in os.listdir(util.NEGATIVE_PATH)
                 if os.path.isfile(os.path.join(util.NEGATIVE_PATH, f))]

numWords = []

for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Positive files finished')

for nf in negativeFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

total_num_Files = len(numWords)

assert total_num_Files == util.NEGATIVE_PATH + util.NUM_POSITIVE_FILES
print('The total number of files is', total_num_Files)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


# Visualize this data in a histogram format using Matplot library
import matplotlib.pyplot as plt
plt.hist(numWords, util.VOCAB_EMB_DIM)
plt.xlabel('Sequence(sentence) length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()

# From the histogram as well as the average number of words per file, we can safely say that most reviews
# will fall under 250 words, which is the max sequence length value we will set in util.py.

# Let's see how we can take a single file and transform it into our ids matrix.
# This is what one of the reviews looks like in text file format.

fname = positiveFiles[3]    # test with index 3. Can use any value
with open(fname) as f:
    for lines in f:
        print(lines)
        exit()


# Now convert to an ids matrix
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[A-Za-z0-9 ]+]")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

firstFile = np.zeros((util.MAX_SEQ_LENGTH), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = clean_sentences(line)
    split = cleanedLine.split()
    for word in split:
        if indexCounter < util.MAX_SEQ_LENGTH:
            try:
                firstFile[indexCounter] = wordsList.index(word)
            except ValueError:
                firstFile[indexCounter] = util.UNK_ID
        indexCounter += 1

print(firstFile)

# Now, let's do the same for each of our 25,000 reviews. We'll load in the movie
# training set and integerize it to get a 25000 x 250 matrix.
def review_to_id():
    ids = np.zeros((total_num_Files, util.MAX_SEQ_LENGTH), dtype='int32')

    # first 12500 are positive, last 12500 are negative
    for review_file in [positiveFiles, negativeFiles]:
        for fileCounter, single_file in enumerate(review_file):
            with open(single_file, "r") as f:
                indexCounter = 0
                line = f.readline()
                cleanedLine = clean_sentences(line)
                split = cleanedLine.split()
                for word in split:
                    if indexCounter < util.MAX_SEQ_LENGTH:
                        try:
                            ids[fileCounter][indexCounter] = wordsList.index(word)
                        except ValueError:
                            ids[fileCounter][indexCounter] = util.UNK_ID
                        indexCounter += 1
                    else:
                        break
                fileCounter += 1

    # Pass into embedding function and see if it evaluates
    np.save('idsMatrix', ids)

# Load pre-computed IDs matrix
ids = np.load('idsMatrix.npy')


# Helper functions for training
from random import randint

def get_train_batch():
    labels = []
    arr = np.zeros([util.BATCH_SIZE, util.MAX_SEQ_LENGTH])
    for i in range(util.BATCH_SIZE):
        # append one positive and one negative review so 12 positive and negative reviews
        # in each batch of size 24
        if i%2 == 0:
            num = randint(1, 11499)     # positive reviews
            labels.append([1,0])
        else:
            num = randint(13499,24999)  # negative reviews
            labels.append([0, 1])
        arr[i] = ids[num-1:num]     # DEBUG: check this
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([util.BATCH_SIZE, util.MAX_SEQ_LENGTH])
    for i in range(util.BATCH_SIZE):
        num = randint(11499, 13499)
        if num <= 12499:    # Last 1000 of positive reviews
            labels.append([1,0])
        else:               # First 1000 of negative reviews
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels



