
# coding: utf-8

# In[1]:

import tensorflow as tf
from train_rnn_last import get_batches
from layers import layer, fork, residual, conv2d, layer2d, pool, pool_frames, gru, gru_last, flatten, flatten_multi, dense, dense_multi, weights, bias, get_summaries
import threading
from itertools import tee
import json
import numpy as np
import datetime

# In[2]:

#from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils


# In[3]:

# In[4]:

def log(message):
    print('{0} {1}'.format(datetime.datetime.now(), message), flush=True)

#def get_pretrained_vars():
#    pretrained_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/hand") + tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/main") + tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/lstm_encoder")
#    missing_vars = ["network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Bias/Adam",
#                    "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Matrix/Adam",
#                    "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Bias/Adam",
#                    "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Matrix/Adam"]
#    return [v for v in pretrained_vars if v.name not in missing_vars]

def get_pretrained_vars():
    pretrained_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/hand") + tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/main") + tf.get_collection(tf.GraphKeys.VARIABLES, scope="network/lstm_encoder")
    return [v for v in pretrained_vars if "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Bias/Adam" not in v.name and "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Matrix/Adam" not in v.name and "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Bias/Adam" not in v.name and "network/lstm_encoder/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Matrix/Adam" not in v.name]

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

def classification_network(x_h, x_m, y, test=False):
    with tf.variable_scope("network") as sc:
        if test:
            sc.reuse_variables()

        with tf.variable_scope("hand"):
            flat_h = network_arm(x_h)
        with tf.variable_scope("main"):
            flat_m = network_arm(x_m)

        combined = tf.concat(2, [flat_h, flat_m])
        flat = gru_last(combined, 512 * 2, 1, batch_size, "lstm_encoder")

        with tf.variable_scope("penultimate"):
            penultimate = dense(flat, 1024, 10, 0.1, 0.02)

        with tf.variable_scope("out"):
            b_output = bias(20, 0.1)
            w_output = weights([10, 20], 0.02)

        output = tf.matmul(penultimate, w_output) + b_output
        return output, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y))

# In[5]:

with open("config.json", 'r') as f:
    config = json.loads(f.read())

log("Creating computation graph")

seq_len = 32
batch_size = config['batch_size']
CHALAP = config['root']

x_h = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2], name="input_hand")
x_m = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2], name="input_main")
y = tf.placeholder(tf.float32, shape=[None, 20], name="y")

out, loss = classification_network(x_h, x_m, y, test=False)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# In[7]:
log("Initializing variables")

MODEL = "3d-resnet-pretrain-classify"

# Create variables
step = 0
sess = tf.Session()

# Initialize/restore
saver = tf.train.Saver(get_pretrained_vars())
sess.run(tf.initialize_all_variables())
saver.restore(sess, "{}/backup/3d-resnet-predict-5".format(CHALAP))
saver = tf.train.Saver()

summary_writer = tf.train.SummaryWriter("{}/summaries/{}".format(CHALAP, MODEL), sess.graph)

# In[ ]:
log("Starting training")

for epoch in range(20):
    acc = []
    batches = get_batches(CHALAP + "/uber", batch_size, seq_len, 1, 400)
    for i,batch in enumerate(batches):
        step += 1
        ta, _ = sess.run([accuracy, train_step], feed_dict={x_h:batch[0], x_m:batch[1], y:batch[2]})
        acc.append(ta)
        log("step %d, training accuracy %g"%(i, ta))

    log("Done with training, starting validation")
    saver.save(sess, "{}/checkpoints/{}".format(CHALAP, MODEL), global_step=epoch)

    val_batches = get_batches(CHALAP + "/uber", batch_size, seq_len, 401, 470)
    val_acc = []
    for batch in val_batches:
        va = sess.run(accuracy, feed_dict={x_h:batch[0], x_m:batch[1], y:batch[2]})
        val_acc.append(va)

    log("Done with epoch: %d, validation accuracy %g" % (epoch, sum(val_acc) / len(val_acc)))
