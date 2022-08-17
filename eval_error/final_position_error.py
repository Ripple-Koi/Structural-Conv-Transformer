import numpy as np
from scipy import interpolate
# import matplotlib.pyplot as plt
# import random
# from scipy import signal
# from functools import reduce
# from pathlib import Path
# import pickle

# # x y h v_x v_y
# BATCH_SIZE = 64
# FREQUENCY = 10  # always evaluate at 10Hz
# num_vehicles = 6
# dataset_list = ['dataset', 'transformed_dataset', 'inertial_dataset']
# DATASET = dataset_list[1]

# result_path = Path('./result/' + DATASET + '/test')
# result_path.mkdir(parents=True, exist_ok=True)
# result_all_steps = np.load(result_path / 'result_all_steps.npy')
# result_last_step = np.load(result_path / 'result_last_step.npy')

# with open('./' + DATASET + '/testset.pkl', 'rb') as f:
#     valdata = pickle.load(f)  # (batch, frame, feature * vehicle) == (24864, 110, 5 * 6)


# def juggle_and_split(data):
#     # Juggle the dimensions to (batch, frame * vehicle, feature)
#     data = np.reshape(data, (-1, 110, num_vehicles, 5))
#     data = np.reshape(data[:, ::int(10/FREQUENCY), :, :], (-1, 11 * FREQUENCY * num_vehicles, 5))
#     # split 6 for past, 5 for future
#     input_tensor = data[:, :6 * FREQUENCY * num_vehicles, :]
#     target_tensor = data[:, -5 * FREQUENCY * num_vehicles:, :]
#     return input_tensor, target_tensor


# input_tensor_val, target_tensor_val = juggle_and_split(valdata)


def eval(result, input_tensor_val, groundtruth, FREQUENCY, num_vehicles):

    latv_result = result[:, :, -1]  # [sample_size, 5 * FREQUENCY]
    longv_result = result[:, :, -2]  # [sample_size, 5 * FREQUENCY]

    lat_0 = input_tensor_val[:len(latv_result), -num_vehicles, 1]
    long_0 = input_tensor_val[:len(latv_result), -num_vehicles, 0]
    latv_0 = input_tensor_val[:len(latv_result), -num_vehicles, -1]
    longv_0 = input_tensor_val[:len(latv_result), -num_vehicles, -2]

    # lat_val = groundtruth[:len(latv_result), :, 1]  # [sample_size, 50]
    # long_val = groundtruth[:len(latv_result), :, 0]  # [sample_size, 50]
    latv_val = groundtruth[:len(latv_result), :, -1]  # [sample_size, 50]
    longv_val = groundtruth[:len(latv_result), :, -2]  # [sample_size, 50]
    # lankeep_idx = np.load('lanekeep_idx2.npy')

    result = []
    for j in range(len(latv_result)):
        y = np.concatenate([np.expand_dims(latv_0[j], axis=-1), latv_result[j, :]], axis=-1)
        x = np.linspace(0, 5, 1 + 6 * FREQUENCY)
        xnew = np.linspace(0, 5, 61)
        f = interpolate.interp1d(x, y, kind='cubic')
        ynew = f(xnew)
        result.append(ynew)
    latv_result = np.array(result)[:, 1:]

    result = []
    for j in range(len(longv_result)):
        y = np.concatenate([np.expand_dims(longv_0[j], axis=-1), longv_result[j, :]], axis=-1)
        x = np.linspace(0, 5, 1 + 6 * FREQUENCY)
        xnew = np.linspace(0, 5, 61)
        f = interpolate.interp1d(x, y, kind='cubic')
        ynew = f(xnew)
        result.append(ynew)
    longv_result = np.array(result)[:, 1:]

    # from v_x v_y
    # latv_delta = latv_result - latv_val[:, 1:]
    # longv_delta = longv_result - longv_val[:, 1:]
    # laterror_v = latv_delta.cumsum(axis=-1) * 0.1
    # longerror_v = longv_delta.cumsum(axis=-1) * 0.1
    # from x y
    laterror = abs(latv_result - latv_val)
    laterror = laterror.cumsum(axis=-1) * 0.1
    longerror = abs(longv_result - longv_val)
    longerror = longerror.cumsum(axis=-1) * 0.1
    # print(laterror)
    # print(longerror)

    # latME_v = abs(laterror_v).mean(axis=0)[9::10]
    # longME_v = abs(longerror_v).mean(axis=0)[9::10]
    latME = laterror.mean(axis=0)[9::10]
    longME = longerror.mean(axis=0)[9::10]
    # lat_std = np.std(laterror, axis=0)[9::10]
    # long_std = np.std(longerror, axis=0)[9::10]

    laterror_all = np.concatenate((np.mean(laterror[:, 0:10], axis=-1)[:, np.newaxis],
                                np.mean(laterror[:, 0:20], axis=-1)[:, np.newaxis],
                                np.mean(laterror[:, 0:30], axis=-1)[:, np.newaxis],
                                np.mean(laterror[:, 0:40], axis=-1)[:, np.newaxis],
                                np.mean(laterror[:, 0:50], axis=-1)[:, np.newaxis],
                                np.mean(laterror[:, 0:60], axis=-1)[:, np.newaxis]), axis=-1)
    longerror_all = np.concatenate((np.mean(longerror[:, 0:10], axis=-1)[:, np.newaxis],
                                    np.mean(longerror[:, 0:20], axis=-1)[:, np.newaxis],
                                    np.mean(longerror[:, 0:30], axis=-1)[:, np.newaxis],
                                    np.mean(longerror[:, 0:40], axis=-1)[:, np.newaxis],
                                    np.mean(longerror[:, 0:50], axis=-1)[:, np.newaxis],
                                    np.mean(longerror[:, 0:60], axis=-1)[:, np.newaxis]), axis=-1)
    latRMSE = np.sqrt((laterror_all ** 2).mean(axis=0))
    longRMSE = np.sqrt((longerror_all ** 2).mean(axis=0))

    return np.stack((latME, longME, latRMSE, longRMSE))

# error_all_steps = eval(result_all_steps, input_tensor_val, target_tensor_val, num_vehicles)
# error_last_step = eval(result_last_step, input_tensor_val, target_tensor_val, num_vehicles)
# np.savetxt(result_path / 'error_all_steps.csv', error_all_steps, delimiter=',')
# np.savetxt(result_path / 'error_last_step.csv', error_last_step, delimiter=',')