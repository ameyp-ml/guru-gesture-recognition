
# coding: utf-8

# In[1]:

import tensorflow as tf
from train_variable_rnn_last import get_batches
from layers import layer, conv2d, layer2d, pool, pool_frames, lstm_variable_last, flatten, flatten_multi, dense, dense_multi, weights, bias, get_summaries


# In[2]:

#from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils


# In[3]:

from tensorflow.contrib.layers.python.layers import batch_norm


# In[4]:

def res_layer(inp, num_features1, stride):
    num_features2 = num_features1 * 4
    shape = inp.get_shape()
    [seq_len, inp_width, num_channels] = [int(shape[i]) for i in [1, 2, 4]]
    #[_, seq_len, inp_width, _, num_channels] = [int(i) for i in list(inp.get_shape())]

    inputs = tf.reshape(inp, [-1, inp_width, inp_width, num_channels])

    if num_channels == num_features2:
        o_l = inputs
    else:
        b_l = bias(num_features2, 0.2)
        w_l = weights([1, 1, num_channels, num_features2], 0.04)
        o_l = conv2d(inputs, b_l, w_l, stride)

    b1_r = bias(num_features1, 0.2)
    w1_r = weights([1, 1, num_channels, num_features1], 0.04)
    conv1_r = tf.nn.relu(batch_norm(conv2d(inputs, b1_r, w1_r, stride)))

    b2_r = bias(num_features1, 0.2)
    w2_r = weights([3, 3, num_features1, num_features1], 0.04)
    conv2_r = tf.nn.relu(batch_norm(conv2d(conv1_r, b2_r, w2_r, 1)))

    b3_r = bias(num_features2, 0.2)
    w3_r = weights([1, 1, num_features1, num_features2], 0.04)
    conv3_r = conv2d(conv2_r, b3_r, w3_r, 1)

    out = tf.nn.relu(batch_norm(tf.add(o_l, conv3_r)))

    shape = out.get_shape()
    [out_width, out_features] = [int(shape[i]) for i in [1, 3]]
    #[_, out_width, _, out_features] = [int(i) for i in list(out.get_shape())]

    return tf.reshape(out, [-1, seq_len, out_width, out_width, out_features])

# In[5]:

print("Creating computation graph", flush=True)

seq_len = 100
batch_size = 2

x_h = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])
x_m = tf.placeholder(tf.float32, shape=[None, seq_len, 64, 64, 2])
y = tf.placeholder(tf.float32, shape=[None, 20])

# Convolutional layers, hand
b_conv1_h, w_conv1_h, h_conv1_h, o_conv1_h = layer(x_h, 16, [5, 5])
o_res1_h = res_layer(o_conv1_h, 16, 1)
o_res2_h = res_layer(o_res1_h, 16, 1)
o_res3_h = res_layer(o_res2_h, 32, 2)
o_res4_h = res_layer(o_res3_h, 32, 1)
o_res5_h = res_layer(o_res4_h, 32, 1)
o_res6_h = res_layer(o_res5_h, 64, 2)
o_res7_h = res_layer(o_res6_h, 64, 1)
o_res8_h = res_layer(o_res7_h, 64, 1)
_, _, _, o_h = layer(o_res8_h, 1, [1,1])
#flat_h = flatten(tf.squeeze(o_h))

flat_h = flatten_multi(o_h)
#b_fc1_h, w_fc1_h, h_fc1_h = dense_multi(flat_h, int(flat_h.get_shape()[2]), 256, 0.1, 0.02)

# Convolutional layers, main
b_conv1_m, w_conv1_m, h_conv1_m, o_conv1_m = layer(x_m, 16, [5, 5])
o_res1_m = res_layer(o_conv1_m, 16, 1)
o_res2_m = res_layer(o_res1_m, 16, 1)
o_res3_m = res_layer(o_res2_m, 32, 2)
o_res4_m = res_layer(o_res3_m, 32, 1)
o_res5_m = res_layer(o_res4_m, 32, 1)
o_res6_m = res_layer(o_res5_m, 64, 2)
o_res7_m = res_layer(o_res6_m, 64, 1)
o_res8_m = res_layer(o_res7_m, 64, 1)
_, _, _, o_m = layer(o_res8_m, 1, [1,1])
#flat_m = flatten(tf.squeeze(o_m))

flat_m = flatten_multi(o_m)
#b_fc1_m, w_fc1_m, h_fc1_m = dense_multi(flat_m, int(flat_m.get_shape()[2]), 256, 0.1, 0.02)

#combined = tf.concat(1, [flat_h, flat_m])
combined = tf.concat(2, [flat_h, flat_m])


# In[6]:

#output = combined

flat = lstm_variable_last(x_m, combined, 512 * 2, 2, batch_size, "lstm")
b_output = bias(20, 0.1)
w_output = weights([1024, 20], 0.02)
output = tf.nn.softmax(tf.matmul(flat, w_output) + b_output)

cross_entropy = -tf.reduce_sum(y * tf.log(output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[7]:
print("Initializing variables", flush=True)

CHALAP = "/home/aparulekar-ms/chalap"
MODEL = "3d-resnet-bn-mono"

# Create variables
step = 0
sess = tf.InteractiveSession()
summary_writer = tf.train.SummaryWriter("{}/summaries/{}".format(CHALAP, MODEL), sess.graph)
saver = tf.train.Saver()
accuracy_summary = tf.placeholder(tf.float32, [])

# Initialize/restore
sess.run(tf.initialize_all_variables())
#saver.restore(sess, "{}/checkpoints/{}/checkpoint-19".format(CHALAP, MODEL))

# Create summary tensors
accuracy_summary_op = tf.scalar_summary("train_accuracy", accuracy_summary)


# In[ ]:
print("Starting training", flush=True)

for epoch in range(20):
    acc = []
    batches = get_batches(batch_size, seq_len, 1, 400)
    for i,batch in enumerate(batches):
        step += 1
        train_accuracy, _ = sess.run([accuracy, train_step], feed_dict={x_h:batch[0], x_m:batch[1], y: batch[2]})
        acc.append(train_accuracy)
        if i%100 == 0:
            cumulative = sum(acc) / len(acc)
            print("step %d, training accuracy %g"%(i, cumulative), flush=True)
            summary_writer.add_summary(accuracy_summary_op.eval(feed_dict={accuracy_summary: cumulative}), step)

    saver.save(sess, "{}/checkpoints/{}".format(CHALAP, MODEL), global_step=epoch)

    val_batches = get_batches(batch_size, seq_len, 401, 470)
    val_cumulative = 0
    val_acc = []
    for j,batch in enumerate(val_batches):
        val_accuracy = sess.run(accuracy, feed_dict={x_h:batch[0], x_m:batch[1], y: batch[2]})
        val_acc.append(val_accuracy)

    print("Done with epoch: %d, validation accuracy %g" % (epoch, sum(val_acc) / len(val_acc)), flush=True)
