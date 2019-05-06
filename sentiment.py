import tensorflow as tf
import numpy as np

import model
import util
import data

import argparse
import os
import re
import sys

def train():
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(util.ITERATIONS):
        # Load in batch of reviews and associated labels
        next_batch, next_batch_labels = data.get_train_batch()

        # fetches argument defines the value we’re interested in computing. We want our
        # optimizer to be computed since it minimizes our loss function
        sess.run(fetches=model.optimizer, feed_dict={model.input_data: next_batch,
                                                     model.labels: next_batch_labels})

        # Write summary to Tensorboard
        if i % 50 == 0:
            summary = sess.run(model.merged, {model.input_data: next_batch,
                                              model.labels: next_batch_labels})
            model.writer.add_summary(summary, i)  # stores loss and accuracy after 50 steps

        # Save the network every 10,000 training iterations
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, util.CP_PATH, global_step=i)
            print("--Saved to %s" % save_path)
    model.writer.close()

    """ Track training progress

    Note that you can track its progress using TensorBoard. While the following cell is running,
    use your terminal to enter the directory that contains this notebook, enter 
        - tensorboard --logdir=tensorboard, 
    and visit http://localhost:6006/ with a browser to keep an eye on your training progress.
    """


def load_pretrained_model():
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    return sess


def evaluation():
    iterations = 10
    sess = load_pretrained_model()
    for i in range(iterations):
        next_batch, next_batch_labels = data.get_test_batch()
        print("Accuracy for this batch:", (sess.run(model.accuracy,
                                                    {model.input_data: next_batch,
                                                     model.labels: next_batch_labels})) * 100)

def _get_user_input():
    """ Get user's input which will be transformed into encoder input later """
    print("> ", end="")
    # sys.stdout.write("> ")
    sys.stdout.flush()
    return sys.stdin.readline()


def get_sentence_matrix(sentence):
    arr = np.zeros([util.BATCH_SIZE, util.MAX_SEQ_LENGTH])
    sentenceMatrix = np.zeros([util.BATCH_SIZE, util.MAX_SEQ_LENGTH], dtype='int32')
    cleanedSentence = data.clean_sentences(sentence)
    split = cleanedSentence.split()
    for indexCounter, word in enumerate(split):
        try:
            sentenceMatrix[0, indexCounter] = data.wordsList.index(word)
        except ValueError:
            sentenceMatrix[0, indexCounter] = util.UNK_ID  # Vector for unkown words
    return sentenceMatrix


def test():
    """ Testing a pretrained network """

    # First create our Tensorflow graph.
    # Start by setting some of our hyperparamters.
    num_dimensions = 300

    # Load in data structures
    wordsList = data.wordsList  # encode words as UTF-8 (necessary?)
    wordVectors = data.wordVectors

    model.build_graph()

    saver = tf.train().Saver()
    with tf.InteractiveSession as sess:
        saver.restore(sess, tf.train().latest_checkpoint('models'))

        max_length = util.MAX_SEQ_LENGTH
        print('Write a movie review to analyse. Enter to exit. Max length is ', max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                continue
            if line == '':
                break

            """
                Before we input our own text, let's first define a couple of functions.
                The first is a function to make sure the sentence is in the proper format,
                and the second is a function that obtains the word vectors for each of the words
                in a given sentence.
            """

            # Get token-ids for the input sentence
            input_matrix = get_sentence_matrix(line)
            if len(input_matrix) > max_length:
                print('Max length I can handle is:', max_length)
                line = _get_user_input()
                continue

            predicted_sentiment = sess.run(model.prediction, {model.input_data: input_matrix})

            if predicted_sentiment[0] > predicted_sentiment[1]:
                print("Positive statement")
            else:
                print("Negative statement")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'test'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    """
    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)
    """

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


if __name__ == '__main__':
    main()