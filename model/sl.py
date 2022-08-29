import tensorflow as tf


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
        self,
        target_x,
        front_x,
        leftfront_x,
        rightfront_x,
        leftrear_x,
        rightrear_x,
        training,
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

        return (
            target_out,
            front_out,
            leftfront_out,
            rightfront_out,
            leftrear_out,
            rightrear_out,
        )


class structural_conv_lstm(tf.keras.Model):
    def __init__(self, d_model, num_layers, frequency, rate, activation, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)

        self.frequency = frequency
        self.num_layers = num_layers
        self.sl_layers = [
            structural_lstm_layer(d_model, rate, activation) for _ in range(num_layers)
        ]

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
        x = self.dense0(inputs)  # (batch_size, inp_seq_len, d_model/4)
        x = self.dropout1(x, training=training)

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
                target_x,
                front_x,
                leftfront_x,
                rightfront_x,
                leftrear_x,
                rightrear_x,
                training,
            )

        x = self.rnn(
            tf.concat(
                [target_x, front_x, leftfront_x, rightfront_x, leftrear_x, rightrear_x],
                axis=-1,
            )
        )  # (batch_size, seq_len, 6 * d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred
