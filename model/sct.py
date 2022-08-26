import numpy as np
import tensorflow as tf


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


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, activation):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # call: mha(v, k, q, mask)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
            ])

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
                 maximum_position_encoding, rate, activation):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation=activation)

        # self.pos_encoding.shape == (1, maximum_position_encoding, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # self.sptl_encoding.shape == (num_vehicles, d_model)
        self.sptl_encoding = spatial_encoding(self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, activation)
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


# TODO: use tf.SparseTensor
def create_interaction_mask(seq_len_1, seq_len_2, num_vehicles):
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


def create_masks(inp, num_vehicles):
    # Encoder padding mask
    # tf.maximum() broadcast: (batch_size, 1, tar_seq_len, tar_seq_len)
    # 'scaled_attention_logits += mask' broadcast: (batch_size, num_heads, inp_seq_len, inp_seq_len)
    enc_padding_mask = create_padding_mask(inp)  # (batch_size, 1, 1, inp_seq_len)
    interaction_mask = create_interaction_mask(tf.shape(inp)[1], tf.shape(inp)[1], num_vehicles)  # (inp_seq_len, inp_seq_len)
    enc_padding_mask = tf.maximum(interaction_mask, enc_padding_mask)

    return enc_padding_mask


class ConvEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, filters, activation, num_vehicles, frequency):
        super(ConvEncoderLayer, self).__init__()
        
        self.filters = filters
        self.num_vehicles = num_vehicles
        self.frequency = frequency

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 9], activation=activation, input_shape=[None, 24, 72, 1])
        self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 4))
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 5], activation=activation)
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
        x = tf.reshape(x, [-1, 5 * self.frequency * self.num_vehicles, self.filters])

        return x  # (batch_size, input_seq_len, filters)


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, activation, num_vehicles, frequency):
        super(ConvEncoder, self).__init__()
        
        self.global_layer = ConvEncoderLayer(filters, activation, num_vehicles, frequency)
        self.medium_layer = ConvEncoderLayer(filters, activation, num_vehicles, frequency)
        self.local_layer = ConvEncoderLayer(filters, activation, num_vehicles, frequency)

    def call(self, graph):
        global_output = self.global_layer(graph[:, 0, :, :, :, tf.newaxis])
        medium_output = self.medium_layer(graph[:, 1, :, :, :, tf.newaxis])
        local_output = self.local_layer(graph[:, 2, :, :, :, tf.newaxis])

        return global_output, medium_output, local_output


class structural_conv_transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 pe_input, rate, activation, num_vehicles, frequency):
        super().__init__()
        self.num_vehicles = num_vehicles
        
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_input, rate, activation)
        self.conv_encoder = ConvEncoder(int(d_model / 4), activation, num_vehicles, frequency)
        self.gru2 = tf.keras.layers.GRU(d_model, return_sequences=False, activation=activation)
        self.dense1 = tf.keras.layers.Dense(6 * frequency)
        self.dense2 = tf.keras.layers.Dense(6 * frequency)

    def call(self, inputs, graph, training):
        # TODO: Keras models prefer if you pass all your inputs (include caches) in the first argument
        # inputs = tf.math.multiply(inputs, tf.constant([1/30, 1/30]))

        x0 = self.dense0(inputs)
        x1, x2, x3 = self.conv_encoder(graph)
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        enc_padding_mask = create_masks(x, self.num_vehicles)
        x = self.encoder(x, training, enc_padding_mask, self.num_vehicles)  # (batch_size, inp_seq_len, d_model)
        x = self.gru2(x[:, ::self.num_vehicles, :])
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)
        
        return long_pred, lat_pred