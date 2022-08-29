import tensorflow as tf


class ConvEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, filters, rate, activation, num_vehicles, frequency):
        super(ConvEncoderLayer, self).__init__()

        self.filters = filters
        self.num_vehicles = num_vehicles
        self.frequency = frequency

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[3, 9],
            activation=activation,
            input_shape=[None, 24, 72, 1],
        )
        self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 4))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=[3, 5], activation=activation
        )
        self.pooling2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 6))
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dense = tf.keras.layers.Dense(filters)

    def call(self, x, training):
        x = tf.reshape(
            x, [-1, 24, 72, 1]
        )  # output [batch_size * seq_len, 22, 64, filters]
        x = tf.cast(x, tf.float32) / 127.5 - 1  # (0, 255) -> (-1, 1)
        x = self.conv1(x)  # output [batch_size * seq_len, 22, 64, filters]
        x = self.pooling1(x)  # output [batch_size * seq_len, 11, 16, filters]
        x = self.conv2(x)  # output [batch_size * seq_len, 9, 12, filters]
        x = self.pooling2(x)  # output [batch_size * seq_len, 3, 2, filters]
        x = self.dropout(x, training=training)
        x = self.dense(x)  # output [batch_size * seq_len, 3, 2, filters]
        x = tf.stack(
            [
                x[:, 1, 0, :],
                x[:, 1, 1, :],
                x[:, 0, 1, :],
                x[:, 2, 1, :],
                x[:, 0, 0, :],
                x[:, 2, 0, :],
            ],
            axis=1,
        )  # output [batch_size * seq_len, 6, filters]
        x = tf.reshape(
            x[:, : self.num_vehicles, :],
            [-1, 5 * self.frequency * self.num_vehicles, self.filters],
        )

        return x  # (batch_size, input_seq_len, filters)


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, rate, activation, num_vehicles, frequency):
        super(ConvEncoder, self).__init__()

        self.global_layer = ConvEncoderLayer(
            filters, rate, activation, num_vehicles, frequency
        )
        self.medium_layer = ConvEncoderLayer(
            filters, rate, activation, num_vehicles, frequency
        )
        self.local_layer = ConvEncoderLayer(
            filters, rate, activation, num_vehicles, frequency
        )

    def call(self, graph, training):
        global_output = self.global_layer(graph[:, 0, :, :, :, tf.newaxis], training)
        medium_output = self.medium_layer(graph[:, 1, :, :, :, tf.newaxis], training)
        local_output = self.local_layer(graph[:, 2, :, :, :, tf.newaxis], training)

        return global_output, medium_output, local_output


class structural_lstm_layer(tf.keras.layers.Layer):
    # |   |   |   |
    # | 2 | 1 | 3 |
    # |   |   |   |
    # |   | 0 |   |
    # |   |   |   |
    # | 4 |   | 5 |
    # |   |   |   |

    def __init__(self, d_model, rate, activation):
        super(structural_lstm_layer, self).__init__()

        self.target_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        self.front_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        self.leftfront_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        self.rightfront_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        self.leftrear_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        self.rightrear_lstm = tf.keras.layers.LSTM(
            d_model, return_sequences=True, activation=activation
        )
        
        self.layernorm0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout0 = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        self.dropout5 = tf.keras.layers.Dropout(rate)

    def call(
        self, target_x, front_x, leftfront_x, rightfront_x, leftrear_x, rightrear_x, training
    ):
        target_inp = tf.concat(
            [target_x, front_x, leftfront_x, rightfront_x, leftrear_x, rightrear_x],
            axis=-1,
        )
        target_out = self.target_lstm(target_inp)
        target_out = self.dropout0(target_out, training=training)
        target_out = self.layernorm0(target_x + target_out)
        
        front_inp = tf.concat([target_x, front_x, leftfront_x, rightfront_x], axis=-1)
        front_out = self.front_lstm(front_inp)
        front_out = self.dropout1(front_out, training=training)
        front_out = self.layernorm1(front_x + front_out)
        
        leftfron_inp = tf.concat([target_x, front_x, leftfront_x, leftrear_x], axis=-1)
        leftfront_out = self.leftfront_lstm(leftfron_inp)
        leftfront_out = self.dropout2(leftfront_out, training=training)
        leftfront_out = self.layernorm2(leftfront_x + leftfront_out)
        
        rightfront_inp = tf.concat(
            [target_x, front_x, rightfront_x, rightrear_x], axis=-1
        )
        rightfront_out = self.rightfront_lstm(rightfront_inp)
        rightfront_out = self.dropout3(rightfront_out, training=training)
        rightfront_out = self.layernorm3(rightfront_x + rightfront_out)
        
        leftrear_inp = tf.concat([target_x, leftfront_x, leftrear_x], axis=-1)
        leftrear_out = self.leftrear_lstm(leftrear_inp)
        leftrear_out = self.dropout4(leftrear_out, training=training)
        leftrear_out = self.layernorm4(leftrear_x + leftrear_out)
        
        rightrear_inp = tf.concat([target_x, rightfront_x, rightrear_x], axis=-1)
        rightrear_out = self.rightrear_lstm(rightrear_inp)
        rightrear_out = self.dropout5(rightrear_out, training=training)
        rightrear_out = self.layernorm5(rightrear_x + rightrear_out)
        
        return target_out, front_out, leftfront_out, rightfront_out, leftrear_out, rightrear_out


class structural_conv_lstm(tf.keras.Model):
    def __init__(self, d_model, num_layers, frequency, rate, activation, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.conv_encoder = ConvEncoder(
            filters=int(d_model / 4),
            rate=rate,
            activation=activation,
            num_vehicles=6,
            frequency=frequency,
        )

        self.frequency = frequency
        self.num_layers = num_layers
        self.sl_layers = [structural_lstm_layer(d_model, rate, activation) for _ in range(num_layers)]

        if rnn_cell == "gru":
            self.rnn = tf.keras.layers.GRU(
                d_model, return_sequences=False, activation=activation
            )
        if rnn_cell == "lstm":
            self.rnn = tf.keras.layers.LSTM(
                d_model, return_sequences=False, activation=activation
            )
        if rnn_cell == "rnn":
            self.rnn = tf.keras.layers.SimpleRNN(
                d_model, return_sequences=False, activation=activation
            )
        self.dense1 = tf.keras.layers.Dense(6 * frequency)
        self.dense2 = tf.keras.layers.Dense(6 * frequency)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, graph, training):
        x0 = self.dense0(inputs)  # (batch_size, inp_seq_len, d_model/4)
        x0 = self.dropout1(x0, training=training)
        x1, x2, x3 = self.conv_encoder(
            graph, training
        )  # (batch_size, inp_seq_len, d_model/4)
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # (batch_size, inp_seq_len, d_model)

        target_x = x[:, :: self.frequency, :]
        front_x = x[:, 1 :: self.frequency, :]
        leftfront_x = x[:, 2 :: self.frequency, :]
        rightfront_x = x[:, 3 :: self.frequency, :]
        leftrear_x = x[:, 4 :: self.frequency, :]
        rightrear_x = x[:, 5 :: self.frequency, :]
        for i in range(self.num_layers):
            (
                target_x,
                front_x,
                leftfront_x,
                rightfront_x,
                leftrear_x,
                rightrear_x,
            ) = self.sl_layers[i](
                target_x, front_x, leftfront_x, rightfront_x, leftrear_x, rightrear_x, training
            )

        x = self.rnn(
            tf.concat([target_x, front_x, leftfront_x, rightfront_x, leftrear_x, rightrear_x], axis=-1)
        )  # (batch_size, seq_len, 6 * d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred
