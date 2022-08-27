# import logging
# import pickle
# import collections
# import os
# import re
# import string
# import sys
# from math import pi
# from datetime import datetime
# from scipy import stats
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model.sct import structural_conv_transformer
from utils import (
    CustomSchedule,
    evaluation,
    loss_function,
    error_function,
    juggle_and_split,
    interpolate_pred,
)

# logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

BATCH_SIZE = 64
EPOCHS = 10
FREQUENCY = 10
NUM_VEHICLES = 6
dataset_list = ["dataset", "transformed_dataset", "inertial_dataset"]
DATASET = "/" + dataset_list[1]
MODEL = "/sct" + f"/{FREQUENCY}Hz"
ACTIVATION = "relu"
LR = 0.0001
LR_SCHEDULE = False

checkpoint_path = Path("./ckpt/" + DATASET + MODEL + "/test")
checkpoint_path.mkdir(exist_ok=True, parents=True)
result_path = Path("./result" + DATASET + MODEL + "/test")
result_path.mkdir(parents=True, exist_ok=True)

data_dir = "./data/val/val.npz"
graph = np.stack(
    [
        np.load(data_dir)["GLOBAL"][:, :: int(10 / FREQUENCY), :, :],
        np.load(data_dir)["MEDIUM"][:, :: int(10 / FREQUENCY), :, :],
        np.load(data_dir)["LOCAL"][:, :: int(10 / FREQUENCY), :, :],
    ],
    axis=1,
)
traj = np.load(data_dir)["TRANSFORM"][
    :, :, [3, 4, 8, 9, 13, 14, 18, 19, 23, 24, 28, 29]
].astype("float32")
graph_train, graph_test, traj_train, traj_test = train_test_split(
    graph, traj, train_size=19200, random_state=42
)
input_traj_train, target_traj_train = juggle_and_split(traj_train, NUM_VEHICLES)
input_traj_test, target_traj_test = juggle_and_split(traj_test, NUM_VEHICLES)

TRAIN_STEPS = len(graph_train) // BATCH_SIZE
TEST_STEPS = len(graph_test) // BATCH_SIZE

predictor = structural_conv_transformer(
    num_layers=1,
    d_model=32,
    num_heads=4,
    dff=32,
    pe_input=5 * FREQUENCY,
    rate=0,
    activation=ACTIVATION,
    num_vehicles=NUM_VEHICLES,
    frequency=FREQUENCY,
)

if LR_SCHEDULE:
    learning_rate = CustomSchedule(
        peak_lr=LR, end_lr=LR / 5, total_steps=EPOCHS * TRAIN_STEPS
    )
else:
    learning_rate = LR
optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

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
    error_lat, error_long = error_function(
        tar, tf.stack([long_pred, lat_pred], axis=-1)
    )
    train_error_lat(error_lat)
    train_error_long(error_long)
    return tf.stack(
        [long_pred, lat_pred], axis=-1
    )  # (batch_size, seq_len, feature_size)


loss_log = []
speed_log = []
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_error_lat = tf.keras.metrics.Mean(name="train_error_lat")
train_error_long = tf.keras.metrics.Mean(name="train_error_long")
for epoch in range(EPOCHS):

    start = time.time()

    graph_train, input_traj_train, target_traj_train = shuffle(
        graph_train, input_traj_train, target_traj_train
    )
    train_loss.reset_states()
    train_error_lat.reset_states()
    train_error_long.reset_states()

    for step in range(TRAIN_STEPS):

        train_step(
            input_traj_train[
                step * BATCH_SIZE : (step + 1) * BATCH_SIZE, :: int(10 / FREQUENCY), :
            ],
            target_traj_train[
                step * BATCH_SIZE : (step + 1) * BATCH_SIZE,
                :: int(10 / FREQUENCY) * NUM_VEHICLES,
                :,
            ],
            graph_train[step * BATCH_SIZE : (step + 1) * BATCH_SIZE, :, :, :, :],
        )
        loss_log.append(
            [epoch * TRAIN_STEPS + step, tf.get_static_value(train_loss.result())]
        )

        if (step + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} Step {step + 1} Loss {train_loss.result():.4f}")

    print(f"Epoch {epoch + 1} Loss {train_loss.result():.4f}")
    print(f"Training time taken for 1 epoch: {time.time() - start:.2f} secs\n")

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

    start_infer = time.time()

    lat_error = []
    long_error = []
    preds = []
    metrics = []
    for step in range(TEST_STEPS):  # Infer in batch to avoid OOM
        train_error_lat.reset_states()
        train_error_long.reset_states()

        # TODO timestep index slice should be [9::10] for 1Hz
        pred = infer_step(
            input_traj_test[
                step * BATCH_SIZE : (step + 1) * BATCH_SIZE, :: int(10 / FREQUENCY), :
            ],
            target_traj_test[
                step * BATCH_SIZE : (step + 1) * BATCH_SIZE,
                :: int(10 / FREQUENCY) * NUM_VEHICLES,
                :,
            ],
            graph_test[step * BATCH_SIZE : (step + 1) * BATCH_SIZE, :, :, :, :],
        )

        interp_pred = interpolate_pred(
            pred, input_traj_test[step * BATCH_SIZE : (step + 1) * BATCH_SIZE, -1:, :]
        )
        batch_metrics = evaluation(
            interp_pred,
            target_traj_test[
                step * BATCH_SIZE : (step + 1) * BATCH_SIZE, ::NUM_VEHICLES, :
            ],
        )

        lat_error.append(train_error_lat.result())
        long_error.append(train_error_long.result())
        preds.append(pred)
        metrics.append(batch_metrics)

    infer_time = (time.time() - start_infer) / (step + 1)
    speed_log.append([epoch + 1, infer_time])
    print(f"Mean inferring time taken of {step + 1} batches: {infer_time:.6f} secs")
    print(
        f"Epoch {epoch + 1} Long Error {np.mean(np.array(long_error)):.4f} Lat Error {np.mean(np.array(lat_error)):.4f}"
    )
    with np.printoptions(formatter={"float": "{: 0.3f}".format}):
        print(np.mean(np.array(metrics), axis=0))

np.savetxt(
    result_path / "metrics.csv", np.mean(np.array(metrics), axis=0), delimiter=","
)

plt.figure()
plt.plot(
    [loss_log[i][0] for i in range(len(loss_log))],
    [loss_log[i][1] for i in range(len(loss_log))],
)
plt.xlabel("Training Steps"), plt.ylabel("Training Loss")
plt.savefig(result_path / "training_loss.jpg")

plt.figure()
plt.plot(
    [speed_log[i][0] for i in range(len(speed_log))],
    [speed_log[i][1] for i in range(len(speed_log))],
)
plt.xlabel("Training Epoch"), plt.ylabel("Inferring Time")
plt.savefig(result_path / "inferring_time.jpg")
