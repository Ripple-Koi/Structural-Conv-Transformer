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
# from sklearn.model_selection import train_test_split
# from scipy import stats

from eval_error import final_position_error

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


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


# n, d = 1000, 64
# pos_encoding = positional_encoding(n, d)
# print(pos_encoding.shape)
# pos_encoding = pos_encoding[0]
#
# # Juggle the dimensions for the plot
# pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
# pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
# pos_encoding = tf.reshape(pos_encoding, (d, n))
# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.ylabel('Depth')
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()

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
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
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


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)  # call: mha(v, k, q, mask)
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # call: mha(v, k, q, mask)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask, num_vehicles):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # look_ahead_mask.shape == (batch_size, 1, target_seq_len, inp_seq_len)
        # padding_mask.shape == (batch_size, 1, 1, inp_seq_len)

        if training:
            attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)

            attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

            ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

            return out3, attn_weights_block1, attn_weights_block2

        else:  # inferring (training=False), only predict the last frame to save time

            # IMPORTANT: look_ahead_mask and padding_mask will broadcast to shape:
            # [batch_size, num_layers, seq_len_q, seq_len_k]
            # thus they should slice the 3rd dimensions to one-step size,
            # because we query only the last frame in each iteration (seq_len_q == 1 * num_vehicles).
            # But the 4th dimensions keep the full size because always look back to key of whole history
            # (seq_len_k == seq_len_inp)

            # x and look_ahead_mask sizes grow with inferring iteration
            # confirm that their shapes match
            # i.e., 'scaled_attention_logits += (mask * -1e9)'can broadcast
            assert x.shape[1] == look_ahead_mask.shape[-1]

            # attn1.shape == (batch_size, 1 * num_vehicles, d_model)
            attn1, attn_weights_block1 = self.mha1(x, x, x[:, -num_vehicles:, :],
                                                   look_ahead_mask[:, :, -num_vehicles:, :])
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x[:, -num_vehicles:, :])  # (batch_size, 1 * num_vehicles, d_model)

            attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask[:, :, -num_vehicles:, :])  # (batch_size, 1 * num_vehicles, d_model)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)  # (batch_size, 1 * num_vehicles, d_model)

            ffn_output = self.ffn(out2)  # (batch_size, 1 * num_vehicles, d_model)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)  # (batch_size, 1 * num_vehicles, d_model)

            return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation='tanh')

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


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation='tanh')

        # self.pos_encoding.shape == (1, maximum_position_encoding, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # self.sptl_encoding.shape == (num_vehicles, d_model)
        self.sptl_encoding = spatial_encoding(self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    # add new parameter 'cache'
    def call(self, x, enc_output, cache, training,
             look_ahead_mask, padding_mask, num_vehicles):
        seq_len = tf.cast(tf.shape(x)[1] / num_vehicles, tf.int32)
        attention_weights = {}

        # adding embedding and position encoding.
        x = self.dense(x)  # (batch_size, input_seq_len, d_x) -> (batch_size, input_seq_len, d_model)
        # pos_encoding is the same for each vehicle
        x += tf.repeat(self.pos_encoding[:, :seq_len, :], tf.cast(num_vehicles * tf.ones(seq_len), dtype=tf.int32),
                       axis=1)
        # sptl_encoding is the same for each frame
        x += tf.tile(self.sptl_encoding, [seq_len, 1])

        x = self.dropout(x, training=training)

        if training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                       look_ahead_mask, padding_mask, num_vehicles)

                # block1.shape == (batch_size, num_heads, target_seq_len, target_seq_len)
                # block2.shape == (batch_size, num_heads, target_seq_len, input_seq_len)
                attention_weights[f'decoder_layer{i + 1}_block1'] = block1
                attention_weights[f'decoder_layer{i + 1}_block2'] = block2

            # x.shape == (batch_size, target_seq_len, d_model)
            return x, attention_weights, None

        else:  # inferring (training=False)
            layer_cache = []

            # x and look_ahead_mask sizes grow with inferring iteration
            # confirm that their shapes match
            # i.e., 'scaled_attention_logits += (mask * -1e9)'can broadcast
            assert x.shape[1] == look_ahead_mask.shape[-1]

            for i in range(self.num_layers):
                # x.shape: (batch_size, target_seq_len, d_model) -> (batch_size, 1 * num_vehicles, d_model)
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                       look_ahead_mask, padding_mask, num_vehicles)
                layer_cache.append(x)

                # After each layer, concat the last frame to
                # the previous sequence when inferring 2nd, 3rd, ... frame
                if cache is not None:
                    # cache.shape == (num_layers, batch_size, target_seq_len - 1 * num_vehicles, d_model)
                    # x.shape: (batch_size, 1 * num_vehicles, d_model) -> (batch_size, target_seq_len, d_model)
                    x = tf.concat([cache[i], x], axis=1)

                # block1.shape == (batch_size, num_heads, 1, target_seq_len)
                # block2.shape == (batch_size, num_heads, 1, input_seq_len)
                attention_weights[f'decoder_layer{i + 1}_block1'] = block1
                attention_weights[f'decoder_layer{i + 1}_block2'] = block2

            # in case of inferring 2nd, 3rd, ... frame
            if cache is not None:
                # (num_layers, batch_size, (target_seq_len - 1 * num_vehicles) + 1 * num_vehicles, d_model)
                cache = tf.concat([cache, tf.stack(layer_cache, axis=0)], axis=2)

            # in case of inferring 1st frame
            else:
                # to be fed as cache when inferring the 2nd frame,
                # so cache.shape[2] is target_seq_len - 1 * num_vehicles (2 * num_vehicles - 1 * num_vehicles)
                cache = tf.stack(layer_cache, axis=0)  # (num_layers, batch_size, 1 * num_vehicles, d_model)

            # x.shape == (batch_size, target_seq_len, d_model)
            return x, attention_weights, cache


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


def create_masks(inp, tar):
    # Encoder padding mask
    # tf.maximum() broadcast: (batch_size, 1, tar_seq_len, tar_seq_len)
    # 'scaled_attention_logits += mask' broadcast: (batch_size, num_heads, inp_seq_len, inp_seq_len)
    enc_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, inp_seq_len)
    interaction_mask = create_interaction_mask(tf.shape(inp)[1], tf.shape(inp)[1])  # (inp_seq_len, inp_seq_len)
    enc_padding_mask = tf.maximum(interaction_mask, enc_padding_mask)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs (k and v).
    # tf.maximum() broadcast: (batch_size, 1, tar_seq_len, inp_seq_len)
    # 'scaled_attention_logits += mask' broadcast: (batch_size, num_heads, tar_seq_len, inp_seq_len)
    # however in inferring target one by one, broadcast: (batch_size, num_heads, 1 * num_vehicles, inp_seq_len)
    # thus need to slice dec_padding_mask in the infer part of 'class Decoder' to match dimensions
    dec_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, inp_seq_len)
    interaction_mask = create_interaction_mask(tf.shape(tar)[1], tf.shape(inp)[1])  # (tar_seq_len, inp_seq_len)
    dec_padding_mask = tf.maximum(interaction_mask, dec_padding_mask)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # (tar_seq_len, tar_seq_len)
    interaction_mask = create_interaction_mask(tf.shape(tar)[1], tf.shape(tar)[1])  # (tar_seq_len, tar_seq_len)
    dec_target_padding_mask = create_padding_mask(tar)  # (batch_size, 1, 1, tar_seq_len)
    # tf.maximum() broadcast: (batch_size, 1, tar_seq_len, tar_seq_len)
    # 'scaled_attention_logits += mask' broadcast: (batch_size, num_heads, tar_seq_len, tar_seq_len)
    look_ahead_mask = tf.maximum(interaction_mask, look_ahead_mask)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 pe_input, pe_target, target_size, rate):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_size)

    def call(self, inputs, cache, training, num_vehicles):
        # TODO: Keras models prefer if you pass all your inputs (include caches) in the first argument
        inp, tar = inputs
        inp = tf.math.multiply(inp, tf.constant(([1/300, 1/300, 1/pi, 1/30, 1/30])))
        tar = tf.math.multiply(tar, tf.constant(([1/300, 1/300, 1/pi, 1/30, 1/30])))
        enc_output_cache, dec_cache = cache

        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        # x and look_ahead_mask sizes grow with inferring iteration
        # confirm that their shapes match
        # i.e., 'scaled_attention_logits += (mask * -1e9)'can broadcast
        assert tar.shape[1] == look_ahead_mask.shape[-1]

        # encode only once in inferring to save time
        if enc_output_cache is None:
            enc_output_cache = self.encoder(inp, training, enc_padding_mask,
                                            num_vehicles)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, dec_cache = self.decoder(
            tar, enc_output_cache, dec_cache, training, look_ahead_mask, dec_padding_mask, num_vehicles)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_size)
        # final_output = tf.math.multiply(final_output, tf.constant(([300, 300, pi, 30, 30])))
        
        return final_output, attention_weights, [enc_output_cache, dec_cache]


# TODO: network parameters
num_layers = 2
d_model = 128
dff = 64
num_heads = 8
dropout_rate = 0.1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=128, warmup_steps=4000):  # total training steps around 70000
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model * 2) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule()
optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# temp_learning_rate_schedule = CustomSchedule()
# plt.plot(temp_learning_rate_schedule(tf.range(70000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

# TODO: 1000 or less
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    pe_input=1000,
    pe_target=1000,
    target_size=5,  # vehicle has 5 features
    rate=dropout_rate)

# TODO: weighted MAE
loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_error_lat = tf.keras.metrics.Mean(name='train_error_lat')
train_error_long = tf.keras.metrics.Mean(name='train_error_long')


def loss_function(real, pred):
    # # loss_.shape == (batch_size, tar_seq_len)
    # loss_ = loss_object(real, pred)

    weights = np.ones(real.shape)
    weights[:, ::num_vehicles, :] = [1, 1, 10, 100, 100]
    loss_ = tf.math.reduce_mean(abs(real - pred) * weights, axis=-1)

    # 0 element is masked with True
    mask = tf.math.equal(real, 0)
    # All-True vector is masked with 0
    mask = tf.math.logical_not(tf.math.reduce_all(mask, axis=-1))
    mask = tf.cast(mask, dtype=tf.float32)  # (batch_size, tar_seq_len)

    loss_ *= mask  # (batch_size, tar_seq_len)

    return tf.math.reduce_sum(loss_) / tf.math.reduce_sum(mask)  # (1, )


# from (x, y, head, vx, vy) watch v_lat v_long
def error_function(real, pred, num_vehicles):
    error_lat = abs(real[:, ::num_vehicles, 4] - pred[:, ::num_vehicles, 4])  # (batch_size, tar_seq_len / num_vehicles)
    error_long = abs(
        real[:, ::num_vehicles, 3] - pred[:, ::num_vehicles, 3])  # (batch_size, tar_seq_len / num_vehicles)

    # 0 element is masked with True
    mask = tf.math.equal(real[:, ::num_vehicles, :], 0)
    # All-True vector is masked with 0
    mask = tf.math.logical_not(tf.math.reduce_all(mask, axis=-1))
    mask = tf.cast(mask, dtype=tf.float32)  # (batch_size, tar_seq_len / num_vehicles)

    error_lat *= mask  # (batch_size, tar_seq_len / num_vehicles)
    error_long *= mask  # (batch_size, tar_seq_len / num_vehicles)

    # TODO: Watch errors in all frames
    return tf.math.reduce_sum(error_lat, axis=0) / tf.math.reduce_sum(mask, axis=0), \
        tf.math.reduce_sum(error_long, axis=0) / tf.math.reduce_sum(mask, axis=0)
    # return tf.math.reduce_sum(error_lat) / tf.math.reduce_sum(mask), tf.math.reduce_sum(error_long) / tf.math.reduce_sum(mask)


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
def train_step(inp, tar, num_vehicles):
    tar_inp = tf.concat([inp[:, -num_vehicles:, :], tar[:, :-num_vehicles, :]], axis=1)
    tar_real = tar[:, :, :]

    with tf.GradientTape() as tape:
        predictions, _, _ = transformer([inp, tar_inp], [None, None], training=True, num_vehicles=num_vehicles)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)


@tf.function
def infer_step(inp, tar, num_vehicles):
    # the last frame of inp (current trajectory), shape: (batch size, 1 * num_vehicles, feature size)
    tar_inp = inp[:, -num_vehicles:, :]
    tar_real = tar[:, :, :]
    cache = [None, None]
    for i in range(int(tar_real.shape[1] / num_vehicles)):
        predictions, _, cache = transformer([inp, tar_inp], cache, training=False, num_vehicles=num_vehicles)

        # select the last frame from the tar_seq_len dimension
        last = predictions[:, -num_vehicles:, :]  # (batch_size, 1 * num_vehicles, feature size)
        # concatenate the last frame to tar_inp which is given to the decoder as its input.
        tar_inp = tf.concat([tar_inp, last], axis=1)

        # uncomment the follows to check look_ahead_mask working (accuracy is not so good.....)
        # tf.debugging.assert_equal(tf.cast(tar_inp[:, num_vehicles:, :], tf.float16),
        #                           tf.cast(predictions[:, :, :], tf.float16), summarize=300)

    error_lat, error_long = error_function(tar_real, tar_inp[:, num_vehicles:, :], num_vehicles)
    train_error_lat(error_lat)
    train_error_long(error_long)
    # all_steps, last_step
    return tar_inp[:, num_vehicles::num_vehicles, :], predictions[:, ::num_vehicles, :]  # (batch_size, seq_len, feature_size)


BATCH_SIZE = 64
EPOCHS = 10
FREQUENCY = 1
num_vehicles = 6
dataset_list = ['dataset', 'transformed_dataset', 'inertial_dataset']
DATASET = '/' + dataset_list[0]
MODEL = '/st' + f'/{FREQUENCY}Hz'

checkpoint_path = Path('./ckpt' + DATASET + MODEL + '/test')
checkpoint_path.mkdir(exist_ok=True, parents=True)
ckpt = tf.train.Checkpoint(transformer=transformer,
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
    data = np.reshape(data, (-1, 110, num_vehicles, 5))
    data = np.reshape(data[:, ::int(10/FREQUENCY), :, :], (-1, 11 * FREQUENCY * num_vehicles, 5))
    # split 6 for past, 5 for future
    input_tensor = data[:, :6 * FREQUENCY * num_vehicles, :]
    target_tensor = data[:, -5 * FREQUENCY * num_vehicles:, :]
    return input_tensor, target_tensor


data = []
with open('.' + DATASET + '/trainset_0.pkl', 'rb') as f:
    data.append(pickle.load(f))  # (batch, frame, feature) == (100000, 110, 5 * num_vehicles)
with open('.' + DATASET + '/trainset_1.pkl', 'rb') as f:
    data.append(pickle.load(f))  # (batch, frame, feature) == (98847, 110, 5 * num_vehicles)
data = np.concatenate((data[0], data[1]), axis=0)
input_tensor_train, target_tensor_train = juggle_and_split(data)  # (batch, frame, feature) == (198847, 110 * num_vehicles, 5)
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

with open('.' + DATASET + '/testset.pkl', 'rb') as f:
    valdata = pickle.load(f)  # (batch, frame, feature) == (24864, 110, 5 * num_vehicles)
input_tensor_val, target_tensor_val = juggle_and_split(valdata)  # (batch, frame, feature) == (24864, 110 * num_vehicles, 5)
groundtruth = valdata[:, -50:, :5]

loss_log = []
speed_log = []
for epoch in range(EPOCHS):

    start = time.time()

    train_loss.reset_states()
    train_error_lat.reset_states()
    train_error_long.reset_states()

    for (step, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
        train_step(inp, tar, num_vehicles)
        loss_log.append([epoch * steps_per_epoch + step, tf.get_static_value(train_loss.result())])

        if step % 100 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')
    print(f'Training time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    
    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
    
    # test on inferring one random sample or all samples
    start_infer = time.time()

    # rand = np.random.randint(input_tensor_val.shape[0])
    # infer_step(input_tensor_val[rand:rand+1, :, :], target_tensor_val[rand:rand+1, :, :], num_vehicles)
    # print(f'Epoch {epoch + 1} Error {train_error_lat.result():.4f} {train_error_long.result():.4f}')
    # print(f'Inferring time for No. {rand} sample: {time.time() - start_infer:.6f} secs')

    lat_error = []
    long_error = []
    result_all_steps = []
    result_last_step = []
    for i in range(len(input_tensor_val) // BATCH_SIZE):
        train_error_lat.reset_states()
        train_error_long.reset_states()

        pred1, pred2 = infer_step(input_tensor_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :],
                                  target_tensor_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :], num_vehicles)

        lat_error.append(train_error_lat.result())
        long_error.append(train_error_long.result())

        result_all_steps.append(pred1)
        result_last_step.append(pred2)
        # print(f'Lat Error {train_error_lat.result():.4f} Long Error {train_error_long.result():.4f}\n')

    lat_error = np.array(lat_error)
    long_error = np.array(long_error)
    infer_time = (time.time() - start_infer) / (i + 1)
    speed_log.append([epoch + 1, infer_time])
    print(f'Mean inferring time taken for {i + 1} batches: {infer_time:.6f} secs')
    print(f'Epoch {epoch + 1} Lat Error {lat_error.mean():.4f} Long Error {long_error.mean():.4f}\n')

plt.figure()
plt.plot([loss_log[i][0] for i in range(len(loss_log))], [loss_log[i][1] for i in range(len(loss_log))])
plt.xlabel('Training Steps'), plt.ylabel('Training Loss')
plt.savefig(result_path / 'training_loss.jpg')

plt.figure()
plt.plot([speed_log[i][0] for i in range(len(speed_log))], [speed_log[i][1] for i in range(len(speed_log))])
plt.xlabel('Training Epoch'), plt.ylabel('Inferring Time')
plt.savefig(result_path / 'inferring_time.jpg')

result_all_steps = np.concatenate(result_all_steps, axis=0)
result_last_step = np.concatenate(result_last_step, axis=0)
print(f'result_all_steps shape: {result_all_steps.shape} result_last_step shape {result_last_step.shape}')
error_all_steps = final_position_error.eval(result_all_steps, input_tensor_val, groundtruth, FREQUENCY, num_vehicles)
error_last_step = final_position_error.eval(result_last_step, input_tensor_val, groundtruth, FREQUENCY, num_vehicles)
np.savetxt(result_path / 'error_all_steps.csv', error_all_steps, delimiter=',')
np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print(error_all_steps)
    print(error_last_step)


