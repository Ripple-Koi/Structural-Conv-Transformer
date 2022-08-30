def model_builder(cfg):
    if cfg.model == "sct":
        from model.sct import structural_conv_transformer

        model = structural_conv_transformer(
            num_layers=1,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dff=cfg.d_model,
            pe_input=5 * cfg.frequency,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            num_vehicles=cfg.num_vehicles,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "st":
        from model.st import structural_transformer

        model = structural_transformer(
            num_layers=1,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dff=cfg.d_model,
            pe_input=5 * cfg.frequency,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            num_vehicles=cfg.num_vehicles,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "scl":
        from model.scl import structural_conv_lstm

        model = structural_conv_lstm(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "sl":
        from model.sl import structural_lstm

        model = structural_lstm(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "clstm":
        from model.baseline import conv_lstm

        model = conv_lstm(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "lstm":
        from model.baseline import lstm_model

        model = lstm_model(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )
        
    if cfg.model == "cgru":
        from model.baseline import conv_gru

        model = conv_gru(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    if cfg.model == "gru":
        from model.baseline import gru_model

        model = gru_model(
            num_layers=2,
            d_model=cfg.d_model,
            rate=cfg.dropout_rate,
            activation=cfg.activation,
            frequency=cfg.frequency,
            rnn_cell=cfg.rnn_cell,
        )

    return model
