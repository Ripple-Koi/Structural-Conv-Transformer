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
from sklearn.model_selection import train_test_split
# from scipy import stats

from eval_error import final_position_error

BATCH_SIZE = 64
EPOCHS = 10
FREQUENCY = 10
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


data_dir = './data/val/val.npz'
graph = np.stack([np.load(data_dir)['GLOBAL'], np.load(data_dir)['MEDIUM'], np.load(data_dir)['LOCAL']], axis=1) 
traj = np.load(data_dir)['TRANSFORM'][:, :, 3:5].astype('float32')
graph_train, graph_test, traj_train, traj_test = train_test_split(graph, traj, train_size=19200, random_state=42)


# result_last_step = target_tensor_val[:, -6 * FREQUENCY:, :]
# print(groundtruth-result_last_step)
input_traj = traj_test[:, :50, :]
target_traj = traj_test[:, 50:, :]
result_last_step = np.repeat(input_traj[:, -1:, :], 6 * FREQUENCY, axis=1)
error_lat = abs(target_traj[:, ::int(10/FREQUENCY), 1] - result_last_step[:, :, 1])  # (batch_size, tar_seq_len)
error_long = abs(target_traj[:, ::int(10/FREQUENCY), 0] - result_last_step[:, :, 0])  # (batch_size, tar_seq_len)
print(np.mean(error_lat))
print(np.mean(error_long))

print(f'result_last_step shape {result_last_step.shape}')
error_last_step = final_position_error.eval(result_last_step, input_traj, target_traj, FREQUENCY, num_vehicles)
np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print(error_last_step)
