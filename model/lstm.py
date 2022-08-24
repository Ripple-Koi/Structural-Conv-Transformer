import logging
import time
import pickle
# import collections
import os
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
num_vehicles = 1
dataset_list = ['dataset', 'transformed_dataset', 'inertial_dataset']
DATASET = '/' + dataset_list[1]
MODEL = '/lstm' + f'/{FREQUENCY}Hz'

import tensorflow as tf

class lstm_model(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True, activation=tf.nn.relu)
    self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False, activation=tf.nn.relu)
    self.dense1 = tf.keras.layers.Dense(6*FREQUENCY)
    self.dense2 = tf.keras.layers.Dense(6*FREQUENCY)

  def call(self, inputs):
    x = self.lstm2(inputs)
    long_pred = self.dense1(x)
    lat_pred = self.dense2(x)
    
    return long_pred, lat_pred

predictor = lstm_model()

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
def train_step(inp, tar):

    with tf.GradientTape() as tape:
        long_pred, lat_pred = predictor(inp)
        loss = loss_function(tar, tf.stack([long_pred, lat_pred], axis=-1))

    gradients = tape.gradient(loss, predictor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))

    train_loss(loss)


@tf.function
def infer_step(inp, tar):
    long_pred, lat_pred = predictor(inp)
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

def split(data):
    data = data[:, ::int(10/FREQUENCY), :]
    # split 5 for past, 6 for future
    input_tensor = data[:, :5 * FREQUENCY, :]
    target_tensor = data[:, -6 * FREQUENCY:, :]
    return input_tensor, target_tensor


data_dir = './data/val/val.npz'
graph = np.stack([np.load(data_dir)['GLOBAL'], np.load(data_dir)['MEDIUM'], np.load(data_dir)['LOCAL']], axis=1) 
traj = np.load(data_dir)['TRANSFORM'][:, :, 3:5].astype('float32')
graph_train, graph_test, traj_train, traj_test = train_test_split(graph, traj, train_size=19200, random_state=42)

TRAIN_STEPS = len(graph_train) // BATCH_SIZE


loss_log = []
speed_log = []
for epoch in range(EPOCHS):

    start = time.time()
    
    graph_train, traj_train = shuffle(graph_train, traj_train)

    train_loss.reset_states()
    train_error_lat.reset_states()
    train_error_long.reset_states()

    for step in range(TRAIN_STEPS):

        train_step(traj_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :50, :], traj_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, 50:, :])
        dummy = graph_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, 0, :, :, :] + graph_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, 1, :, :, :] + graph_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, 2, :, :, :]
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

    train_error_lat.reset_states()
    train_error_long.reset_states()

    pred = infer_step(traj_test[:, :50, :], traj_test[:, 50:, :])

    lat_error = np.array(train_error_lat.result())
    long_error = np.array(train_error_long.result())

    result_last_step = pred
        # print(f'Lat Error {train_error_lat.result():.4f} Long Error {train_error_long.result():.4f}\n')

    infer_time = time.time() - start_infer
    speed_log.append([epoch + 1, infer_time])
    print(f'Mean inferring time taken for 1 batch: {infer_time:.6f} secs')
    print(f'Epoch {epoch + 1} Lat Error {lat_error.mean():.4f} Long Error {long_error.mean():.4f}\n')

plt.figure()
plt.plot([loss_log[i][0] for i in range(len(loss_log))], [loss_log[i][1] for i in range(len(loss_log))])
plt.xlabel('Training Steps'), plt.ylabel('Training Loss')
plt.savefig(result_path / 'training_loss.jpg')

plt.figure()
plt.plot([speed_log[i][0] for i in range(len(speed_log))], [speed_log[i][1] for i in range(len(speed_log))])
plt.xlabel('Training Epoch'), plt.ylabel('Inferring Time')
plt.savefig(result_path / 'inferring_time.jpg')

print(f'result_last_step shape {result_last_step.shape}')
error_last_step = final_position_error.eval(result_last_step, traj_test[:, :50, :], traj_test[:, 50:, :], FREQUENCY, num_vehicles)
np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print(error_last_step)