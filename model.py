"""
Create our Tensorflow graph.
We’ll first need to define some hyperparameters, such as batch size, number of LSTM units,
number of output classes, and number of training iterations.
"""

import datetime
import tensorflow as tf

import util
import data

batch_size = util.BATCH_SIZE
lstm_units = util.LSTM_UNITS
num_classes = 2
iterations = util.ITERATIONS

tf.reset_default_graph()

"""        
class SentimentModel:
    def __init__(self, forward_only, batch_size):
        ''' forward_only: if set we don't construct the backward pass in the model '''
        print("Initialize new model")
        self.fw_only = forward_only
        self.batch_size = batch_size
"""

# specify two placeholders, one for the inputs into the network, and one for the labels.
input_data = tf.placeholder(tf.int32, [batch_size, util.MAX_SEQ_LENGTH])
labels = tf.placeholder(tf.float32, [batch_size, num_classes])

'''
The most important part about defining these placeholders is understanding each of their 
dimensionalities. The labels placeholder represents a set of values, each either [1, 0] or [0, 1],
depending on whether each training example is positive or negative. 

Each row in the integerized input placeholder represents the integerized representation of each 
training example that we include in our batch.
'''

# tf.nn.lookup(): get our word vectors embedding.
data_ = tf.Variable(tf.zeros([batch_size, util.MAX_SEQ_LENGTH, util.VOCAB_EMB_DIM]),
                    dtype=tf.float32)
# returns a 3-D Tensor of dimensionality
#     - batch size by max sequence length by word vector dimensions.
#     - 24 x 250 x 50
data_ = tf.nn.embedding_lookup(data.wordVectors, input_data)

""" Feed this input into an LSTM network """
# tf.nn.rnn_cell.BasicLSTMCell: takes in an integer for the number of LSTM units that we want.
# This is one of the hyperparameters that will take some tuning to figure out the optimal value
cell_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_units)

# Then wrap that LSTM cell in a dropout layer to help prevent the network from overfitting
if not forward_only:    # Train
    cell_lstm = tf.contrib.rnn.DropoutWrapper(cell=cell_lstm, output_keep_prob=0.75)
else:   # Test
    cell_lstm = tf.contrib.rnn.DropoutWrapper(cell=cell_lstm, output_keep_prob=0.25)

# tf.nn.dynamic_rnn: unrolls the whole network and creates a pathway for the data to flow
# through the RNN graph.
# The first output of the dynamic RNN function can be thought of as the last hidden state vector
lstm_output, _ = tf.nn.dynamic_rnn(cell_lstm, data_, dtype=tf.float32)

'''
More advanced network architecture choice of stacking multiple LSTM cells would imporve the 
long term dependency thereby improving the result but also increase training time and number
of model parameters ...
'''

""" Weights and bias for final output layer """
weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

# This vector will be reshaped and then multiplied by a final weight matrix and a bias term to
# obtain the final output values.
lstm_output = tf.transpose(lstm_output, [1, 0, 2])
last_lstm_output = tf.gather(lstm_output, int(lstm_output.get_shape()[0]) - 1)

prediction = (tf.matmul(last_lstm_output, weight) + bias)

""" define correct prediction and accuracy metrics to track how the network is doing """
# The correct prediction formulation works by looking at the index of the maximum value of
# the 2 output values, and then seeing whether it matches with the training labels.
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

""" Loss and back propagation """
# Define a standard cross entropy loss with a softmax layer put on top of the final prediction
# values. Use Adam optimizer with default learning rate of .001.
# maybe tf.nn.softmax_cross_entropy_with_logits_v2 ?????????????????????????????
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

""" Using Tensorboard to visualize the loss and accuracy values """
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()

logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + "/")
writer = tf.summary.FileWriter(logdir, tf.Session().graph)  # sess.graph


def build_graph(self):
    self._create_placeholders()
    self._inference()
    self._create_loss()
    self._create_optimizer()
    self._create_summary()
