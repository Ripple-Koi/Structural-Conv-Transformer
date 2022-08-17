import numpy as np
import pickle
from copy import deepcopy
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# open trainset
data = []
with open('./transformed_dataset/trainset_0.pkl', 'rb') as f:
    data.append(pickle.load(f))  # (batch, frame, feature * vehicle) == (100000, 110, 5 * 6)
with open('./transformed_dataset/trainset_1.pkl', 'rb') as f:
    data.append(pickle.load(f))  # (batch, frame, feature * vehicle) == (98847, 110, 5 * 6)
data = np.concatenate((data[0], data[1]), axis=0)

v_lat = np.mean(deepcopy(data[:,:,4]))  # vy
v_long = np.mean(deepcopy(data[:,:,3]))  # vx
print([v_lat, v_long])
delta_vx = v_long * np.ones(110, dtype='float32')
delta_x = (delta_vx.cumsum(axis=-1) - delta_vx) * 0.1


def add_inertia(data):
    # observing the trajectories in speed of v_lat == 0 and v_long == v
    for frame in range(110):
        for vehicle in range(6):
            if any(data[frame, (0+5*vehicle):(5+5*vehicle)] != 0.):
                # x
                data[frame, 0+5*vehicle] = data[frame, 0+5*vehicle] - delta_x[frame]
                # vx
                data[frame, 3+5*vehicle] = data[frame, 3+5*vehicle] - delta_vx[frame]
    return data


print(delta_vx)
print(delta_x)
with tqdm_joblib(desc='Preprocessing', total=len(data)) as progress_bar:
    outputs = Parallel(n_jobs=8)(delayed(add_inertia)(data[scenario]) for scenario in range(len(data)))

# write trainset
with open('./inertial_dataset/trainset_0.pkl', 'wb') as f:
    pickle.dump(np.array(outputs)[:100000], f)
with open('./inertial_dataset/trainset_1.pkl', 'wb') as f:
    pickle.dump(np.array(outputs)[100000:], f)

# open testset
with open('./transformed_dataset/testset.pkl', 'rb') as f:
    data = pickle.load(f)  # (batch, frame, feature * vehicle) == (24864, 110, 5 * 6)

print(delta_vx)
print(delta_x)
with tqdm_joblib(desc='Preprocessing', total=len(data)) as progress_bar:
    outputs = Parallel(n_jobs=8)(delayed(add_inertia)(data[scenario]) for scenario in range(len(data)))

# write testset
with open('./inertial_dataset/testset.pkl', 'wb') as f:
    pickle.dump(np.array(outputs), f)

