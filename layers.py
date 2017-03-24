import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

def visualize(tensor):
    weights = tensor.eval()
    assert(weights.shape[0] == weights.shape[1])
    width = weights.shape[0]
    channels = weights.shape[2]
    filters = weights.shape[3]

    tiling = math.ceil(math.sqrt(channels * filters))
    output_width = tiling * width
    output = np.zeros((output_width, output_width))
    row, col = 0, 0
    for i in range(channels):
        for j in range(filters):
            output[width*row:width*(row+1),width*col:width*(col+1)] = weights[:,:,i,j]
            col += 1
            if col == tiling:
                col = 0
                row += 1

    return output

def get_summaries(tensor, name):
    channels = tf.split(2, tensor.get_shape()[2], tensor)
    tensors = []
    for t in channels:
        tensors += tf.split(3, int(t.get_shape()[3]) / 16, t)

    return [tf.image_summary("{}/{}".format(name, i), put_kernels_on_grid(w, 4, 4), max_images=1) for (i,w) in enumerate(tensors)]

def extract_last_relevant(outputs):
    """
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
            activations of each in the batch for each time step as returned by
            tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in the
            batch with all later time steps being zeros. Should be of type tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation for
            each example in the batch.
    """
    # Query shape.
    batch_size = int(outputs.get_shape()[0])
    max_length = int(outputs.get_shape()[1])
    num_neurons = int(outputs.get_shape()[2])
    # Index into flattened array as a workaround.
    index = tf.range(0, batch_size) * max_length + (max_length - 1)
    flat = tf.reshape(outputs, [-1, num_neurons])
    relevant = tf.gather(flat, index)
    return relevant

def bias(num_units, init):
    return tf.get_variable("bias", [num_units], initializer=tf.constant_initializer(init))

def weights(shape, stddev):
    return tf.get_variable("weight", shape, initializer=tf.random_normal_initializer(0.0, stddev))

def conv2d(inp, b, w, stride):
    return tf.nn.conv2d(inp, w, strides=[1,stride,stride,1], padding='SAME') + b

def pool2d(inp):
    return tf.nn.max_pool(inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def pool(inp):
    return tf.nn.max_pool3d(inp, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def pool_frames(inp):
    return tf.nn.max_pool3d(inp, ksize=[1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')

def res_layer(inp, num_features1, stride):
    num_features2 = num_features1 * 4
    [batch_size, seq_len, inp_width, _, num_channels] = [int(i) for i in list(inp.get_shape())]

    inputs = tf.reshape(inp, [batch_size * seq_len, inp_width, inp_width, num_channels])

    if num_channels == num_features2:
        o_l = inputs
    else:
        b_l = bias(num_features2, 0.2)
        w_l = weights([1, 1, num_channels, num_features2], 0.04)
        o_l = conv2d(inputs, b_l, w_l, stride)

    b1_r = bias(num_features1, 0.2)
    w1_r = weights([1, 1, num_channels, num_features1], 0.04)
    conv1_r = tf.nn.relu(conv2d(inputs, b1_r, w1_r, stride))

    b2_r = bias(num_features1, 0.2)
    w2_r = weights([3, 3, num_features1, num_features1], 0.04)
    conv2_r = tf.nn.relu(conv2d(conv1_r, b2_r, w2_r, 1))

    b3_r = bias(num_features2, 0.2)
    w3_r = weights([1, 1, num_features1, num_features2], 0.04)
    conv3_r = conv2d(conv2_r, b3_r, w3_r, 1)

    out = tf.nn.relu(batch_norm(tf.add(o_l, conv3_r)))

    [_, out_width, _, out_features] = [int(i) for i in list(out.get_shape())]

    return tf.reshape(out, [batch_size, seq_len, out_width, out_width, out_features])

def layer(inp, num_units, filter_shape):
    num_channels = int(inp.get_shape()[4])
    b = bias(num_units, 0.2)
    w = weights(filter_shape + [num_channels, num_units], 0.04)
    conv = [tf.nn.relu(conv2d(i, b, w, 1)) for i in tf.unpack(inp, axis=1)]
    return tf.pack(conv, axis=1)

def layer2d(inp, num_units, filter_shape):
    num_channels = int(inp.get_shape()[4])
    b = bias(num_units, 0.2)
    w = weights(filter_shape + [num_channels, num_units], 0.04)
    conv = [conv2d(i, b, w) for i in tf.unpack(inp, axis=1)]
    o_pool = tf.pack([pool2d(c) for c in conv], axis=1)
    return o_pool

def dense(inp, num_in, num_out, b_init, w_init):
    b = bias(num_out, b_init)
    w = weights([num_in, num_out], w_init)
    h = tf.nn.relu(batch_norm(tf.matmul(inp, w) + b))

    return h

def dense_multi(inp, num_in, num_out, b_init, w_init):
    b = bias(num_out, b_init)
    w = weights([num_in, num_out], w_init)
    unpacked = tf.unpack(inp, axis=1)
    h = [tf.nn.relu(tf.matmul(i, w) + b) for i in unpacked]
    return tf.pack(h, axis=1)

def flatten(inp):
    shape = inp.get_shape()[1:].num_elements()
    reshaped = tf.reshape(inp, [-1, shape])
    return reshaped

def flatten_multi(inp):
    seq_len = int(inp.get_shape()[1])
    shape = inp.get_shape()[2:].num_elements()
    reshaped = tf.reshape(inp, [-1, seq_len, shape])
    return reshaped

def fork(inp):
    seq_len = int(inp.get_shape()[1])
    slice_len = int(seq_len/2)
    inputs = tf.unpack(inp, axis=1)
    new_inp = tf.pack(inputs[:slice_len], axis=1)
    past = tf.pack(inputs[:slice_len][::-1], axis=1)
    future = tf.pack(inputs[slice_len:], axis=1)
    return new_inp, past, future

def lstm_last(inp, num_units, num_layers, batch_size, name):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    out, _ = tf.nn.rnn(
        cell,
        tf.unpack(inp, axis=1),
        dtype=tf.float32,
        scope=name)

    return out[-1]

def gru(inp, num_units, num_layers, batch_size, name):
    lstm_cell = tf.nn.rnn_cell.GRUCell(num_units)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    out, _ = tf.nn.rnn(
        cell,
        tf.unpack(inp, axis=1),
        dtype=tf.float32,
        scope=name)

    return tf.pack(out, axis=1)

def gru_last(inp, num_units, num_layers, batch_size, name):
    lstm_cell = tf.nn.rnn_cell.GRUCell(num_units)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    out, _ = tf.nn.rnn(
        cell,
        tf.unpack(inp, axis=1),
        dtype=tf.float32,
        scope=name)

    return out[-1]

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[2,3,4]))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def lstm_variable_last(data, inp, num_units, num_layers, batch_size, name):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    interval_size = length(data)
    #initial = cell.zero_state(batch_size, tf.float32)

    out, _ = tf.nn.dynamic_rnn(
        cell,
        inp,
        dtype=tf.float32,
        time_major=False,
        sequence_length=interval_size,
        scope=name)

    return last_relevant(out, interval_size)

def residual(inp, num_features1, stride):
    num_features2 = num_features1 * 4
    shape = inp.get_shape()
    [seq_len, inp_width, num_channels] = [int(shape[i]) for i in [1, 2, 4]]
    #[_, seq_len, inp_width, _, num_channels] = [int(i) for i in list(inp.get_shape())]

    inputs = tf.reshape(inp, [-1, inp_width, inp_width, num_channels])

    if num_channels == num_features2:
        o_l = inputs
    else:
        with tf.variable_scope("layer_left"):
            b_l = bias(num_features2, 0.2)
            w_l = weights([1, 1, num_channels, num_features2], 0.04)
            o_l = conv2d(inputs, b_l, w_l, stride)

    with tf.variable_scope("layer1_right"):
        b1_r = bias(num_features1, 0.2)
        w1_r = weights([1, 1, num_channels, num_features1], 0.04)
        conv1_r = tf.nn.relu(batch_norm(conv2d(inputs, b1_r, w1_r, stride)))

    with tf.variable_scope("layer2_right"):
        b2_r = bias(num_features1, 0.2)
        w2_r = weights([3, 3, num_features1, num_features1], 0.04)
        conv2_r = tf.nn.relu(batch_norm(conv2d(conv1_r, b2_r, w2_r, 1)))

    with tf.variable_scope("layer3_right"):
        b3_r = bias(num_features2, 0.2)
        w3_r = weights([1, 1, num_features1, num_features2], 0.04)
        conv3_r = conv2d(conv2_r, b3_r, w3_r, 1)

    with tf.variable_scope("output"):
        out = tf.nn.relu(batch_norm(tf.add(o_l, conv3_r)))

    shape = out.get_shape()
    [out_width, out_features] = [int(shape[i]) for i in [1, 3]]
    #[_, out_width, _, out_features] = [int(i) for i in list(out.get_shape())]

    return tf.reshape(out, [-1, seq_len, out_width, out_width, out_features])
