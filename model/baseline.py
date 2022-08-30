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


class lstm_model(tf.keras.Model):
    def __init__(self, num_layers, d_model, rate, activation, frequency, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.num_layers = num_layers
        self.lstm_layers = [
            tf.keras.layers.LSTM(d_model, return_sequences=True, activation=activation)
            for _ in range(num_layers)
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
        x = self.dense0(inputs[:, ::6, :])  # (batch_size, seq_len, d_model/4)
        x = self.dropout1(x, training=training)
        for i in range(self.num_layers):
            x = self.lstm_layers[i](x)
        x = self.rnn(x)  # (batch_size, seq_len, d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred


class conv_lstm(tf.keras.Model):
    def __init__(self, num_layers, d_model, rate, activation, frequency, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.conv_encoder = ConvEncoder(
            filters=int(d_model / 4),
            rate=rate,
            activation=activation,
            num_vehicles=6,
            frequency=frequency,
        )
        self.num_layers = num_layers
        self.lstm_layers = [
            tf.keras.layers.LSTM(d_model, return_sequences=True, activation=activation)
            for _ in range(num_layers)
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
        x0 = self.dense0(inputs[:, ::6, :])  # (batch_size, seq_len, d_model/4)
        x0 = self.dropout1(x0, training=training)
        x1, x2, x3 = self.conv_encoder(
            graph, training
        )  # (batch_size, inp_seq_len, d_model/4)
        x = tf.concat(
            [x0, x1[:, ::6, :], x2[:, ::6, :], x3[:, ::6, :]], axis=-1
        )  # (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.lstm_layers[i](x)
        x = self.rnn(x)  # (batch_size, seq_len, d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred


class gru_model(tf.keras.Model):
    def __init__(self, num_layers, d_model, rate, activation, frequency, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.num_layers = num_layers
        self.gru_layers = [
            tf.keras.layers.GRU(d_model, return_sequences=True, activation=activation)
            for _ in range(num_layers)
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
        x = self.dense0(inputs[:, ::6, :])  # (batch_size, seq_len, d_model/4)
        x = self.dropout1(x, training=training)
        for i in range(self.num_layers):
            x = self.gru_layers[i](x)
        x = self.rnn(x)  # (batch_size, seq_len, d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred


class conv_gru(tf.keras.Model):
    def __init__(self, num_layers, d_model, rate, activation, frequency, rnn_cell):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(int(d_model / 4), activation=activation)
        self.conv_encoder = ConvEncoder(
            filters=int(d_model / 4),
            rate=rate,
            activation=activation,
            num_vehicles=6,
            frequency=frequency,
        )
        self.num_layers = num_layers
        self.gru_layers = [
            tf.keras.layers.GRU(d_model, return_sequences=True, activation=activation)
            for _ in range(num_layers)
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
        x0 = self.dense0(inputs[:, ::6, :])  # (batch_size, seq_len, d_model/4)
        x0 = self.dropout1(x0, training=training)
        x1, x2, x3 = self.conv_encoder(
            graph, training
        )  # (batch_size, inp_seq_len, d_model/4)
        x = tf.concat(
            [x0, x1[:, ::6, :], x2[:, ::6, :], x3[:, ::6, :]], axis=-1
        )  # (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.gru_layers[i](x)
        x = self.rnn(x)  # (batch_size, seq_len, d_model)
        x = self.dropout2(x, training=training)
        long_pred = self.dense1(x)
        lat_pred = self.dense2(x)

        return long_pred, lat_pred