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

BATCH_SIZE = 64
EPOCHS = 10
FREQUENCY = 1
num_vehicles = 1
dataset_list = ['dataset', 'transformed_dataset', 'inertial_dataset']
DATASET = '/' + dataset_list[1]
MODEL = '/cv' + f'/{FREQUENCY}Hz'

result_path = Path('./result' + DATASET + MODEL + '/test')
result_path.mkdir(parents=True, exist_ok=True)


def juggle_and_split(data):
    # Juggle the dimensions to (batch, frame * vehicle, feature)
    data = np.reshape(data, (-1, 110, num_vehicles, 5))
    data = np.reshape(data[:, ::int(10/FREQUENCY), :, :], (-1, 11 * FREQUENCY * num_vehicles, 5))
    # split 5 for past, 6 for future
    input_tensor = data[:, :5 * FREQUENCY * num_vehicles, :]
    target_tensor = data[:, -6 * FREQUENCY * num_vehicles:, :]
    return input_tensor, target_tensor


data = []
with open('.' + DATASET + '/trainset_0.pkl', 'rb') as f:
    data.append(pickle.load(f)[:, :, :5])  # (batch, frame, feature) == (100000, 110, 5)
with open('.' + DATASET + '/trainset_1.pkl', 'rb') as f:
    data.append(pickle.load(f)[:, :, :5])  # (batch, frame, feature) == (98847, 110, 5)
data = np.concatenate((data[0], data[1]), axis=0)
input_tensor_train, target_tensor_train = juggle_and_split(data)
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

with open('.' + DATASET + '/testset.pkl', 'rb') as f:
    valdata = pickle.load(f)[:, :, :5]  # (batch, frame, feature) == (24864, 110, 5)
input_tensor_val, target_tensor_val = juggle_and_split(valdata)
groundtruth = valdata[:, -60:, :5]

# result_last_step = target_tensor_val[:, -6 * FREQUENCY:, :]
# print(groundtruth-result_last_step)
result_last_step = np.repeat(input_tensor_val[:, -1:, :], 6 * FREQUENCY, axis=1)
error_lat = abs(groundtruth[:, ::int(10/FREQUENCY), 4] - result_last_step[:, :, 4])  # (batch_size, tar_seq_len)
error_long = abs(groundtruth[:, ::int(10/FREQUENCY), 3] - result_last_step[:, :, 3])  # (batch_size, tar_seq_len)
print(np.mean(error_lat))
print(np.mean(error_long))

print(f'result_last_step shape {result_last_step.shape}')
error_last_step = final_position_error.eval(result_last_step, input_tensor_val, groundtruth, FREQUENCY, num_vehicles)
np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print(error_last_step)
