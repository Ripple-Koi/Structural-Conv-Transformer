import logging
import time
import pickle
# import collections
# import os
# import re
# import string
# import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from math import pi
# from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from scipy import stats

from eval_error import final_position_error

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

BATCH_SIZE = 64
EPOCHS = 10
FREQUENCY = 10
num_vehicles = 6
dataset_list = ['dataset', 'transformed_dataset', 'inertial_dataset']
DATASET = '/' + dataset_list[1]
MODEL = '/sct' + f'/{FREQUENCY}Hz'
ACTIVATION = 'relu'

# TODO: 10000 or less
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)  # (1, position, d_model)


def spatial_encoding(d_model):  # output.shape == (num_vehicles, d_model)
    target_vehicle = np.zeros(d_model)
    front_vehicle = np.zeros(d_model)
    front_vehicle[1::2] = np.linspace(0., 1., int(d_model / 2))
    left_front_vehicle = np.zeros(d_model)
    left_front_vehicle[0::2] = np.linspace(0., -1., int(d_model / 2))
    left_front_vehicle[1::2] = np.linspace(0., 1., int(d_model / 2))
    right_front_vehicle = np.zeros(d_model)
    right_front_vehicle[0::2] = np.linspace(0., 1., int(d_model / 2))
    right_front_vehicle[1::2] = np.linspace(0., 1., int(d_model / 2))
    left_rear_vehicle = np.zeros(d_model)
    left_rear_vehicle[0::2] = np.linspace(0., -1., int(d_model / 2))
    left_rear_vehicle[1::2] = np.linspace(0., -1., int(d_model / 2))
    right_rear_vehicle = np.zeros(d_model)
    right_rear_vehicle[0::2] = np.linspace(0., 1., int(d_model / 2))
    right_rear_vehicle[1::2] = np.linspace(0., -1., int(d_model / 2))

    return tf.cast(tf.stack([target_vehicle, front_vehicle, left_front_vehicle,
                             right_front_vehicle, left_rear_vehicle, right_rear_vehicle]), dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions, i.e., batch_size.
    k, v must have matching penultimate (last second) dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1, i.e., attention "weights".
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):  # (v, k, q) as in figure, not (q, k, v) as in equation
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# TODO: relu or selu or tanh
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=ACTIVATION),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # call: mha(v, k, q, mask)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation=ACTIVATION)

        # self.pos_encoding.shape == (1, maximum_position_encoding, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # self.sptl_encoding.shape == (num_vehicles, d_model)
        self.sptl_encoding = spatial_encoding(self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, num_vehicles):
        seq_len = tf.cast(tf.shape(x)[1] / num_vehicles, tf.int32)

        # adding embedding and position encoding.
        x = self.dense(x)  # (batch_size, input_seq_len, d_x) -> (batch_size, input_seq_len, d_model)
        # pos_encoding is the same for each vehicle
        x += tf.repeat(self.pos_encoding[:, :seq_len, :], tf.cast(num_vehicles * tf.ones(seq_len), dtype=tf.int32),
                       axis=1)
        # sptl_encoding is the same for each frame
        x += tf.tile(self.sptl_encoding, [seq_len, 1])

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


def create_padding_mask(seq):
    # seq.shape == (batch_size, seq_len * num_vehicles, d_x)
    seq = tf.math.equal(seq, 0)  # 0 is masked with True
    # seq.shape == (batch_size, seq_len * num_vehicles)
    seq = tf.cast(tf.math.reduce_all(seq, axis=-1), tf.float32)  # Frame with all true is masked with 1

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len * num_vehicles)


def create_look_ahead_mask(seq_len):
    # lower triangular part is unmasked
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len / num_vehicles, seq_len / num_vehicles)), -1, 0)
    # turn into stairs-like matrix
    mask = tf.repeat(mask, repeats=num_vehicles, axis=0)
    mask = tf.repeat(mask, repeats=num_vehicles, axis=1)

    return mask  # (seq_len * num_vehicles, seq_len * num_vehicles)


# TODO: use tf.SparseTensor
def create_interaction_mask(seq_len_1, seq_len_2):
    # |   |   |   |
    # | 2 | 1 | 3 |
    # |   |   |   |
    # |   | 0 |   |
    # |   |   |   |
    # | 4 |   | 5 |
    # |   |   |   |
    # interactions <1,4> <1,5> <2,3> <4,5> <2,5> <3,4> are masked with 1
    mask = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 1, 0, 1],
                     [0, 0, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 1],
                     [0, 1, 1, 0, 1, 0]])
    # tile into full shape
    mask = tf.tile(mask, [seq_len_1 / num_vehicles, seq_len_2 / num_vehicles])

    return tf.cast(mask, tf.float32)  # (seq_len * num_vehicles, seq_len * num_vehicles)


def create_masks(inp):
    # Encoder padding mask
    # tf.maximum() broadcast: (batch_size, 1, tar_seq_len, tar_seq_len)
    # 'scaled_attention_logits += mask' broadcast: (batch_size, num_heads, inp_seq_len, inp_seq_len)
    enc_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, inp_seq_len)
    interaction_mask = create_interaction_mask(tf.shape(inp)[1], tf.shape(inp)[1])  # (inp_seq_len, inp_seq_len)
    enc_padding_mask = tf.maximum(interaction_mask, enc_padding_mask)

    return enc_padding_mask


class ConvEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvEncoderLayer, self).__init__()
        
        self.filters = filters

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 9], activation=ACTIVATION, input_shape=[None, 24, 72, 1])
        self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 4))
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 5], activation=ACTIVATION)
        self.pooling2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 6))
        self.dense = tf.keras.layers.Dense(filters)

    def call(self, x):
        x = tf.reshape(x, [-1, 24, 72, 1])
        x = tf.cast(x, tf.float32) / 127.5 - 1  # (0, 255) -> (-1, 1)
        x = self.conv1(x)  # output [None, 22, 64, filters]
        x = self.pooling1(x)  # output [None, 11, 16, filters]
        x = self.conv2(x)  # output [None, 9, 12, filters]
        x = self.pooling2(x)  # output [None, 3, 2, filters]
        x = self.dense(x)  # output [None, 3, 2, filters]
        x = tf.stack([x[:, 1, 0, :], x[:, 1, 1, :], x[:, 0, 1, :], x[:, 2, 1, :], x[:, 0, 0, :], x[:, 2, 0, :]], axis=1)  # output [None, num_vehicles, filters]
        x = tf.reshape(x, [-1, 5 * FREQUENCY * num_vehicles, self.filters])

        return x  # (batch_size, input_seq_len, filters)


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvEncoder, self).__init__()
        
        self.global_layer = ConvEncoderLayer(filters)
        self.medium_layer = ConvEncoderLayer(filters)
        self.local_layer = ConvEncoderLayer(filters)

    def call(self, graph):
        global_output = self.global_layer(graph[:, 0, :, :, :, tf.newaxis])
        medium_output = self.medium_layer(graph[:, 1, :, :, :, tf.newaxis])
        local_output = self.local_layer(graph[:, 2, :, :, :, tf.newaxis])

        return global_output, medium_output, local_output


class structural_conv_transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 pe_input, rate):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=ACTIVATION)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_input, rate)
        self.conv_encoder = ConvEncoder(int(d_model / 4))
        self.gru2 = tf.keras.layers.GRU(d_model, return_sequences=False, activation=ACTIVATION)
        self.dense1 = tf.keras.layers.Dense(6*FREQUENCY)
        self.dense2 = tf.keras.layers.Dense(6*FREQUENCY)

    def call(self, inputs, graph, training, num_vehicles):
        # TODO: Keras models prefer if you pass all your inputs (include caches) in the first argument
        # inputs = tf.math.multiply(inputs, tf.constant([1/30, 1/30]))

        x0 = self.dense0(inputs)
        x1, x2, x3 = self.conv_encoder(graph)
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        enc_padding_mask = create_masks(x)
        x = self.encoder(x, training, enc_padding_mask, num_vehicles)  # (batch_size, inp_seq_len, d_model)
        x = self.gru2(x[:, ::num_vehicles, :])
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)
        
        return long_pred, lat_pred


predictor = structural_conv_transformer(num_layers=1, d_model=32, num_heads=4, dff=32, pe_input=5*FREQUENCY, rate=0)


##################### TODO: all below is the same, should bagged into function #####################


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model=128, warmup_steps=4000):  # total training steps around 70000
#         super(CustomSchedule, self).__init__()

#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)

#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)

#         return tf.math.rsqrt(self.d_model * 2) * tf.math.minimum(arg1, arg2)


# learning_rate = CustomSchedule()
optimizer = tf.keras.optimizers.Adam(0.0001)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# temp_learning_rate_schedule = CustomSchedule()
# plt.plot(temp_learning_rate_schedule(tf.range(70000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

# TODO: weighted MAE
loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_error_lat = tf.keras.metrics.Mean(name='train_error_lat')
train_error_long = tf.keras.metrics.Mean(name='train_error_long')


def loss_function(real, pred):

    loss = tf.math.reduce_mean(abs(real - pred))

    return loss

def error_function(real, pred):
    error_lat = abs(real[:, :, 1] - pred[:, :, 1])  # (batch_size, tar_seq_len / num_vehicles)
    error_long = abs(real[:, :, 0] - pred[:, :, 0])  # (batch_size, tar_seq_len / num_vehicles)
    return tf.math.reduce_mean(error_lat, axis=0), tf.math.reduce_mean(error_long, axis=0)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# It seems faster without it.....
# TODO: inp.shape and tar.shape based on vehicle numbers
# train_step_signature = [
#     tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
#     tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
# ]


# @tf.function(input_signature=train_step_signature)
@tf.function
def train_step(inp, tar, graph):

    with tf.GradientTape() as tape:
        long_pred, lat_pred = predictor(inp, graph, num_vehicles=6)
        loss = loss_function(tar, tf.stack([long_pred, lat_pred], axis=-1))

    gradients = tape.gradient(loss, predictor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))

    train_loss(loss)


@tf.function
def infer_step(inp, tar, graph):
    long_pred, lat_pred = predictor(inp, graph, num_vehicles=6)
    error_lat, error_long = error_function(tar, tf.stack([long_pred, lat_pred], axis=-1))
    train_error_lat(error_lat)
    train_error_long(error_long)
    # all_steps, last_step
    return tf.stack([long_pred, lat_pred], axis=-1)  # (batch_size, seq_len, feature_size)


checkpoint_path = Path('./ckpt/' + DATASET + MODEL + '/test')
checkpoint_path.mkdir(exist_ok=True, parents=True)
ckpt = tf.train.Checkpoint(predictor,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')

result_path = Path('./result' + DATASET + MODEL + '/test')
result_path.mkdir(parents=True, exist_ok=True)


def juggle_and_split(data):
    # Juggle the dimensions to (batch, frame * vehicle, feature)
    data = np.reshape(data, (-1, 110, num_vehicles, 2))
    data = np.reshape(data, (-1, 110 * num_vehicles, 2))
    # split 5 for past, 6 for future
    input_tensor = data[:, :50 * num_vehicles, :]
    target_tensor = data[:, -60 * num_vehicles:, :]
    return input_tensor, target_tensor


data_dir = './data/val/val.npz'
graph = np.stack([np.load(data_dir)['GLOBAL'][:, ::int(10 / FREQUENCY), :, :], np.load(data_dir)['MEDIUM'][:, ::int(10 / FREQUENCY), :, :], np.load(data_dir)['LOCAL'][:, ::int(10 / FREQUENCY), :, :]], axis=1) 
traj = np.load(data_dir)['TRANSFORM'][:, :, [3,4,8,9,13,14,18,19,23,24,28,29]].astype('float32')
graph_train, graph_test, traj_train, traj_test = train_test_split(graph, traj, train_size=19200, random_state=42)
input_traj_train, target_traj_train = juggle_and_split(traj_train)
input_traj_test, target_traj_test = juggle_and_split(traj_test)

TRAIN_STEPS = len(graph_train) // BATCH_SIZE
TEST_STEPS = len(graph_test) // BATCH_SIZE


loss_log = []
speed_log = []
for epoch in range(EPOCHS):

    start = time.time()
    
    graph_train, input_traj_train, target_traj_train = shuffle(graph_train, input_traj_train, target_traj_train)

    train_loss.reset_states()
    train_error_lat.reset_states()
    train_error_long.reset_states()

    for step in range(TRAIN_STEPS):

        train_step(input_traj_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, ::int(10 / FREQUENCY), :], target_traj_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, ::int(10 / FREQUENCY) * num_vehicles, :], graph_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :, :, :, :])
        loss_log.append([epoch * TRAIN_STEPS + step, tf.get_static_value(train_loss.result())])

        if step % 10 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')
    print(f'Training time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    # test on inferring one random sample or all samples
    start_infer = time.time()

    # rand = np.random.randint(input_tensor_val.shape[0])
    # infer_step(input_tensor_val[rand:rand+1, :, :], target_tensor_val[rand:rand+1, :, :])
    # print(f'Epoch {epoch + 1} Error {train_error_lat.result():.4f} {train_error_long.result():.4f}')
    # print(f'Inferring time for No. {rand} sample: {time.time() - start_infer:.6f} secs')

    lat_error = []
    long_error = []
    result_last_step = []
    for step in range(TEST_STEPS):  # Infer in batch to avoid OOM
        train_error_lat.reset_states()
        train_error_long.reset_states()

        pred = infer_step(input_traj_test[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, ::int(10 / FREQUENCY), :], target_traj_test[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, ::int(10 / FREQUENCY) * num_vehicles, :], graph_test[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :, :, :, :])

        lat_error.append(train_error_lat.result())
        long_error.append(train_error_long.result())

        result_last_step.append(pred)
        # print(f'Lat Error {train_error_lat.result():.4f} Long Error {train_error_long.result():.4f}\n')

    lat_error = np.array(lat_error)
    long_error = np.array(long_error)
    infer_time = (time.time() - start_infer) / (step + 1)
    speed_log.append([epoch + 1, infer_time])
    print(f'Mean inferring time taken of {step + 1} batches: {infer_time:.6f} secs')
    print(f'Epoch {epoch + 1} Lat Error {lat_error.mean():.4f} Long Error {long_error.mean():.4f}\n')

plt.figure()
plt.plot([loss_log[i][0] for i in range(len(loss_log))], [loss_log[i][1] for i in range(len(loss_log))])
plt.xlabel('Training Steps'), plt.ylabel('Training Loss')
plt.savefig(result_path / 'training_loss.jpg')

plt.figure()
plt.plot([speed_log[i][0] for i in range(len(speed_log))], [speed_log[i][1] for i in range(len(speed_log))])
plt.xlabel('Training Epoch'), plt.ylabel('Inferring Time')
plt.savefig(result_path / 'inferring_time.jpg')

result_last_step = np.concatenate(result_last_step, axis=0)
print(f'result_last_step shape {result_last_step.shape}')
error_last_step = final_position_error.eval(result_last_step, input_traj_test[:, ::int(10 / FREQUENCY), :], target_traj_test[:, ::num_vehicles, :], FREQUENCY, num_vehicles)
np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print(error_last_step)