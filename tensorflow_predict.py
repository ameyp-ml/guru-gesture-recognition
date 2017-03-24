
# coding: utf-8

# In[1]:

import tensorflow as tf
from train_rnn_augment import get_batches
from layers import fork, layer, residual, conv2d, layer2d, pool, pool_frames, gru, gru_last, flatten, flatten_multi, dense, dense_multi, weights, bias, get_summaries
import threading
from itertools import tee
import json
import numpy as np

# In[2]:

#from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils


# In[3]:

# In[4]:

def network_arm(inputs):
    with tf.variable_scope("conv1"):
        o_conv1 = layer(inputs, 16, [5, 5])
    with tf.variable_scope("res1"):
        o_res1 = residual(o_conv1, 16, 1)
    with tf.variable_scope("res2"):
        o_res2 = residual(o_res1, 16, 1)
    with tf.variable_scope("res3"):
        o_res3 = residual(o_res2, 32, 2)
    with tf.variable_scope("res4"):
        o_res4 = residual(o_res3, 32, 1)
    with tf.variable_scope("res5"):
        o_res5 = residual(o_res4, 32, 1)
    with tf.variable_scope("res6"):
        o_res6 = residual(o_res5, 64, 2)
    with tf.variable_scope("res7"):
        o_res7 = residual(o_res6, 64, 1)
    with tf.variable_scope("res8"):
        o_res8 = residual(o_res7, 64, 1)
    with tf.variable_scope("fc"):
        fc = layer(o_res8, 1, [1,1])
    with tf.variable_scope("flat"):
        return flatten_multi(fc)

def prediction_network(x_h, x_m, test=False):
    with tf.variable_scope("network") as sc:
        if test:
            sc.reuse_variables()

        with tf.variable_scope("hand"):
            flat_h = network_arm(x_h)
        with tf.variable_scope("main"):
            flat_m = network_arm(x_m)

        combined = tf.concat(2, [flat_h, flat_m])
        combined, past, future = fork(combined)
        encoded = gru_last(combined, 512 * 2, 1, batch_size, "lstm_encoder")

        decoded_past = gru(combined, 512, 1, batch_size, "lstm_past")
        decoded_future = gru(combined, 512, 1, batch_size, "lstm_future")

        return tf.add(tf.nn.l2_loss(tf.sub(decoded_past, past)),
                      tf.nn.l2_loss(tf.sub(decoded_future, future)))

def classification_network(x_h, x_m, y, test=False):
    with tf.variable_scope("network") as sc:
        if test:
            sc.reuse_variables()

        with tf.variable_scope("hand"):
            flat_h = network_arm(x_h)
        with tf.variable_scope("main"):
            flat_m = network_arm(x_m)

        combined = tf.concat(2, [flat_h, flat_m])
        flat = gru_last(combined, 512 * 2, 2, batch_size, "lstm")

        with tf.variable_scope("out") as sc:
            b_output = bias(20, 0.1)
            w_output = weights([1024, 20], 0.02)

        output = tf.matmul(flat, w_output) + b_output
        return output, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y))

# In[5]:

with open("config.json", 'r') as f:
    config = json.loads(f.read())

print("Creating computation graph", flush=True)

seq_len = 32
batch_size = config['batch_size']
CHALAP = config['root']
TRAIN_SAMPLES = 44199
VAL_SAMPLES = 8315

class PredictionRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, data_iterator):
        self.x_h = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])
        self.x_m = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])

        self.threads = []
        self.data_iterator = data_iterator
        self.init_queue()

    def init_queue(self):
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[seq_len, 64, 64, 2], [seq_len, 64, 64, 2]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=1000,
                                           min_after_dequeue=0)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.x_h, self.x_m])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        x_h_batch, x_m_batch = self.queue.dequeue_up_to(batch_size)
        return x_h_batch, x_m_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for x_h, x_m, y in self.data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.x_h:x_h, self.x_m:x_m})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        self.threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)

class ClassificationRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, data_iterator):
        self.x_h = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])
        self.x_m = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])
        self.y = tf.placeholder(tf.float32, shape=[None, 20])
        self.threads = []
        self.data_iterator = data_iterator
        self.init_queue()

    def init_queue(self):
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[seq_len, 64, 64, 2], [seq_len, 64, 64, 2], [20]],
                                           dtypes=[tf.float32, tf.float32, tf.float32],
                                           capacity=1000,
                                           min_after_dequeue=0)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.x_h, self.x_m, self.y])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        x_h_batch, x_m_batch, y_batch = self.queue.dequeue_up_to(batch_size)
        return x_h_batch, x_m_batch, y_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for x_h, x_m, y in self.data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.x_h:x_h, self.x_m:x_m, self.y:y})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        self.threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)

# Network for prediction
with tf.device("/cpu:0"):
    train_runner = PredictionRunner(lambda : get_batches(CHALAP, batch_size, seq_len, 1, 400, 0, 0, 0, True))
    val_runner = PredictionRunner(lambda : get_batches(CHALAP, batch_size, seq_len, 401, 470, 0, 0, 0, True))
    train_x_h, train_x_m = train_runner.get_inputs()
    val_x_h, val_x_m = val_runner.get_inputs()

train_loss = prediction_network(train_x_h, train_x_m, test=False)
train_step = tf.train.AdamOptimizer(1e-4).minimize(train_loss)

val_loss = prediction_network(val_x_h, val_x_m, test=True)


# Network for classification
"""
with tf.device("/cpu:0"):
    train_runner = ClassificationRunner(lambda : get_batches(CHALAP, batch_size, seq_len, 1, 400, 5, 5, 3, True))
    val_runner = ClassificationRunner(lambda : get_batches(CHALAP, batch_size, seq_len, 401, 470, 0, 0, 0, True))
    train_x_h, train_x_m, train_y = train_runner.get_inputs()
    val_x_h, val_x_m, val_y = val_runner.get_inputs()

train_out, train_loss = prediction_network(train_x_h, train_x_m, train_y, test=False)
val_out, val_loss = prediction_network(val_x_h, val_x_m, val_y, test=True)

train_step = tf.train.AdamOptimizer(1e-4).minimize(train_loss)
train_prediction = tf.equal(tf.argmax(train_out, 1), tf.argmax(train_y, 1))
train_accuracy = tf.reduce_mean(tf.cast(train_prediction, tf.float32))

val_prediction = tf.equal(tf.argmax(val_out, 1), tf.argmax(val_y, 1))
val_accuracy = tf.reduce_mean(tf.cast(val_prediction, tf.float32))
"""

# In[7]:
print("Initializing variables", flush="True")

MODEL = "3d-resnet-predict"

# Create variables
step = 0
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
summary_writer = tf.train.SummaryWriter("{}/summaries/{}".format(CHALAP, MODEL), sess.graph)
saver = tf.train.Saver()
accuracy_summary = tf.placeholder(tf.float32, [])

# Initialize/restore
sess.run(tf.initialize_all_variables())
#saver.restore(sess, "{}/checkpoints/{}/checkpoint-19".format(CHALAP, MODEL))

# Create summary tensors
accuracy_summary_op = tf.scalar_summary("train_accuracy", accuracy_summary)

tf.train.start_queue_runners(sess=sess)
train_runner.start_threads(sess)
val_runner.start_threads(sess)

# In[ ]:
print("Starting training", flush="True")

for epoch in range(20):
    cumulative = 0
    for i in range(5000):
        step += 1
        tloss, _ = sess.run([train_loss, train_step])
        cumulative += tloss
        print("step %d, training loss %g"%(i, tloss), flush=True)

    print("Done with training, training loss %g" % (cumulative), flush=True)

    saver.save(sess, "{}/checkpoints/{}".format(CHALAP, MODEL), global_step=epoch)

    vloss = 0
    for i in range(1663):
        vloss += sess.run(val_loss)

    print("Done with epoch: %d, validation loss %g" % (epoch, vloss), flush=True)

"""
for epoch in range(20):
    acc = []
    for i in range(5000):
        step += 1
        ta, _ = sess.run([train_accuracy, train_step])
        acc.append(ta)
        cumulative = sum(acc) / len(acc)
        print("step %d, training accuracy %g"%(i, cumulative), flush=True)
        if i % 100 == 0:
            summary_writer.add_summary(sess.run(accuracy_summary_op, feed_dict={accuracy_summary: cumulative}), step)
        i += 1

    print("Done with training")

    saver.save(sess, "{}/checkpoints/{}".format(CHALAP, MODEL), global_step=epoch)

    val_cumulative = 0
    val_acc = []
    for i in range(1663):
        val_acc.append(sess.run(val_accuracy))

    print("Done with epoch: %d, validation accuracy %g" % (epoch, sum(val_acc) / len(val_acc)), flush=True)
"""
