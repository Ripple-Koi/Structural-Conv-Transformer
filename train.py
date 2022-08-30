import time
import wandb
import numpy as np
import tensorflow as tf
from pathlib import Path

from model.model_builder import model_builder
from utils import (
    DataGenerator,
    CustomSchedule,
    evaluation,
    loss_function,
    interpolate_pred,
)

# logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
default_configs = dict(
    model="sct",
    dataset="transformed_dataset",
    batch_size=8,
    epochs=30,
    frequency=10,
    num_vehicles=6,
    activation="tanh",
    learning_rate=0.0003,
    lr_schedule=True,
    d_model=1024,
    num_heads=4,
    dropout_rate=0.3,
    rnn_cell="gru",
    num_layers=3,
    # TODO Feature size
    # TODO LR parameters
    # TODO BatchNorm, LayerNorm
    # TODO CNN num, kernel, pool
    # TODO Layer-wise activation, d_model
    # TODO Dataset
)
wandb.init(config=default_configs, project="best-sct", tags=["grid-sweep"])
cfg = wandb.config

checkpoint_path = Path(f"./ckpt/{cfg.dataset}/{cfg.model}/{cfg.frequency}Hz/test")
checkpoint_path.mkdir(exist_ok=True, parents=True)
result_path = Path(f"./result/{cfg.dataset}/{cfg.model}/{cfg.frequency}Hz/test")
result_path.mkdir(parents=True, exist_ok=True)

data_generator = DataGenerator(cfg.frequency, cfg.num_vehicles)

if cfg.model != "cv":
    predictor = model_builder(cfg)

    if cfg.lr_schedule:
        learning_rate = CustomSchedule(
            peak_lr=cfg.learning_rate,
            end_lr=cfg.learning_rate / 5,
            total_steps=cfg.epochs * len(data_generator.graph_train) / cfg.batch_size,
        )
    else:
        learning_rate = cfg.learning_rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    ckpt = tf.train.Checkpoint(predictor=predictor, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')


@tf.function
def train_step(inp, tar, graph):
    with tf.GradientTape() as tape:
        long_pred, lat_pred = predictor(inp, graph, training=True)
        loss = loss_function(tar, tf.stack([long_pred, lat_pred], axis=-1))

    gradients = tape.gradient(loss, predictor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))
    train_loss(loss)


@tf.function
def infer_step(inp, tar, graph):
    long_pred, lat_pred = predictor(inp, graph, training=False)
    loss = loss_function(tar, tf.stack([long_pred, lat_pred], axis=-1))

    val_loss(loss)
    return tf.stack(
        [long_pred, lat_pred], axis=-1
    )  # (batch_size, seq_len, feature_size)


train_loss = tf.keras.metrics.Mean(name="train_loss")
val_loss = tf.keras.metrics.Mean(name="val_loss")
for epoch in range(cfg.epochs):
    data_generator.shuffle_trainset()
    train_loss.reset_states()
    val_loss.reset_states()
    train_loss_log = []
    val_loss_log = []
    # preds = []
    metrics = []

    start = time.time()

    if cfg.model != "cv":

        for step, (inp, tar, graph) in enumerate(
            data_generator.next_train_batch(cfg.batch_size)
        ):

            train_step(inp, tar, graph)
            train_loss_log.append(tf.get_static_value(train_loss.result()))

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1} Step {step + 1} Loss {train_loss.result():.4f}"
                )

        mean_train_loss = np.mean(np.array(train_loss_log))
        mean_train_time = (time.time() - start) / (step + 1)
        print(f"Epoch {epoch + 1} Mean Train Loss {mean_train_loss:.4f}")
        print(f"Mean training time for {step + 1} steps: {mean_train_time:.2f} secs\n")

        if (epoch + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

    start_infer = time.time()

    for step, (inp, tar, graph, current_frame, ground_truth) in enumerate(
        data_generator.next_test_batch(cfg.batch_size)
    ):  # Infer in batch to avoid OOM
        if cfg.model == "cv":
            pred = np.repeat(
                inp[:, -cfg.num_vehicles : (-cfg.num_vehicles + 1), :],
                6 * cfg.frequency,
                axis=1,
            )

        else:
            pred = infer_step(inp, tar, graph)

        interp_pred = interpolate_pred(pred, current_frame)
        batch_metrics = evaluation(interp_pred, ground_truth)

        val_loss_log.append(tf.get_static_value(val_loss.result()))
        # preds.append(pred)
        metrics.append(batch_metrics)

    mean_val_loss = np.mean(np.array(val_loss_log))
    mean_infer_time = (time.time() - start_infer) / (step + 1)
    print(f"Epoch {epoch + 1} Mean Val Loss {mean_val_loss:.4f}")
    print(f"Mean inferring time of {step + 1} steps: {mean_infer_time:.6f} secs")

    metrics_table = np.mean(np.array(metrics), axis=0)
    with np.printoptions(formatter={"float": "{: 0.3f}".format}):
        print(metrics_table)

    wandb.log(
        {
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "train_time": mean_train_time,
            "infer_time": mean_infer_time,
            "final_long_error": metrics_table[0, -1],
            "final_lat_error": metrics_table[1, -1],
            "final_distance_error": metrics_table[2, -1],
            "average_long_error": metrics_table[3, -1],
            "average_lat_error": metrics_table[4, -1],
            "average_distance_error": metrics_table[5, -1],
        }
    )

    if cfg.model == "cv":
        break

np.savetxt(result_path / "metrics.csv", metrics_table, delimiter=",")
