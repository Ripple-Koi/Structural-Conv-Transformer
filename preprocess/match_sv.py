import pandas as pd
import numpy as np
import pickle
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from pathlib import Path
from pandas import DataFrame
from math import pi
from typing import List, Tuple
from numpy import ndarray

def find_sv(relative_xy, vehicles, transformed_vehicles):
    front_x = 15
    leftfront_x = 15
    rightfront_x = 15
    leftrear_x = -15
    rightrear_x = -15
    front_id = None
    leftfront_id = None
    rightfront_id = None
    leftrear_id = None
    rightrear_id = None
    sv = np.zeros(30)
    transformed_sv = np.zeros(30)
    for i in range(len(relative_xy)):
        
        if relative_xy[i, 1] == 0 and relative_xy[i, 0] == 0:
            target_id = vehicles[i, -1]
            target_x = relative_xy[i, 0]
            sv[0:5] = vehicles[i, 0:5]
            transformed_sv[0:5] = transformed_vehicles[i, 0:5]
            continue

        if -1.5 < relative_xy[i, 1] <= 1.5 and 0 < relative_xy[i, 0] <= front_x:
            front_id = vehicles[i, -1]
            front_x = relative_xy[i, 0]
            sv[5:10] = vehicles[i, 0:5]
            transformed_sv[5:10] = transformed_vehicles[i, 0:5]
            continue
        
        if 1.5 < relative_xy[i, 1] <= 4.5 and 0 < relative_xy[i, 0] <= leftfront_x:
            leftfront_id = vehicles[i, -1]
            leftfront_x = relative_xy[i, 0]
            sv[10:15] = vehicles[i, 0:5]
            transformed_sv[10:15] = transformed_vehicles[i, 0:5]
            continue
        
        if -4.5 < relative_xy[i, 1] <= -1.5 and 0 < relative_xy[i, 0] <= rightfront_x:
            rightfront_id = vehicles[i, -1]
            rightfront_x = relative_xy[i, 0]
            sv[15:20] = vehicles[i, 0:5]
            transformed_sv[15:20] = transformed_vehicles[i, 0:5]
            continue
        
        if 1.5 < relative_xy[i, 1] <= 4.5 and leftrear_x < relative_xy[i, 0] <= 0:
            leftrear_id = vehicles[i, -1]
            leftrear_x = relative_xy[i, 0]
            sv[20:25] = vehicles[i, 0:5]
            transformed_sv[20:25] = transformed_vehicles[i, 0:5]
            continue
        
        if -4.5 < relative_xy[i, 1] <= -1.5 and rightrear_x < relative_xy[i, 0] <= 0:
            rightrear_id = vehicles[i, -1]
            rightrear_x = relative_xy[i, 0]
            sv[25:30] = vehicles[i, 0:5]
            transformed_sv[25:30] = transformed_vehicles[i, 0:5]
            continue
    
    return sv, transformed_sv, [target_id, target_x, front_id, front_x, leftfront_id, leftfront_x, \
        rightfront_id, rightfront_x, leftrear_id, leftrear_x, rightrear_id, rightrear_x]


def relative_heading(heading, theta):
    if heading - theta <= -pi:
        return heading - theta + 2*pi
    elif heading - theta > pi:
        return heading - theta - 2*pi
    else:
        return heading - theta


def move_and_rotate(vehicles, theta_0, x_0, y_0):
    vehicles = vehicles.T
    # anti-clockwise rotate the axis for theta so that focal heading is zero
    rotate_matrix = np.array([[np.cos(theta_0), np.sin(theta_0)],[-np.sin(theta_0), np.cos(theta_0)]])
    xy = np.matmul(rotate_matrix, vehicles[0:2, :] - np.array([[x_0], [y_0]]))
    heading = np.array([relative_heading(vehicles[2:3, i], theta_0) for i in range(len(vehicles[0, :]))]).T
    vxvy = np.matmul(rotate_matrix, vehicles[3:5, :])
    transformed_vehicles = np.concatenate([xy, heading, vxvy])
    return transformed_vehicles.T


def match_sv(tracks: DataFrame, traj_save_path: Path) -> Tuple[List[ndarray], List[ndarray]]:    
    # tracks = pd.read_parquet(file)
    focal = tracks[tracks['track_id']==tracks['focal_track_id']]
    features = ['position_x','position_y','heading','velocity_x','velocity_y','track_id']
    x_0 = focal['position_x'].tolist()[0]
    y_0 = focal['position_y'].tolist()[0]
    theta_0 = focal['heading'].tolist()[0]  # -PI < theta <= PI
    sv_traj = []
    transformed_sv_traj = []
    for timestep in range(110):
        x = focal['position_x'].tolist()[timestep]
        y = focal['position_y'].tolist()[timestep]
        theta = focal['heading'].tolist()[timestep]  # -PI < theta <= PI
        rotate_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        
        vehicles = tracks[tracks['timestep']==timestep][features].to_numpy(copy=True)
        transformed_vehicles = move_and_rotate(vehicles, theta_0, x_0, y_0)
        relative_xy = np.matmul(rotate_matrix, vehicles.T[0:2, :] - np.array([[x], [y]])).T
        sv, transformed_sv, _ = find_sv(relative_xy, vehicles, transformed_vehicles)
        sv_traj.append(sv)
        transformed_sv_traj.append(transformed_sv)
    return sv_traj, transformed_sv_traj
    # np.savez_compressed(str(traj_save_path), ORIGIN=sv_traj, TRANSFORM=transformed_sv_traj) 


# if __name__ == "__main__":
#     # sample, transformed_sample = make_sv_list('./train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet')
#     def should_make_list(file):    
#         tracks = pd.read_parquet(file)
#         focal = tracks[tracks['track_id']==tracks['focal_track_id']]
#         return True if len(focal['timestep']) == 110 else False

#     # av2 val -> testset
#     files = list(Path('./val').rglob('*.parquet'))
#     with tqdm_joblib(desc='Preprocessing', total=25000) as progress_bar:
#         outputs = Parallel(n_jobs=8)(delayed(match_sv)(file) for file in files if should_make_list(file))

#     dataset = np.array([output[0] for output in outputs], dtype='float32')
#     print(dataset)
#     print(dataset.shape)
#     with open('./dataset/testset.pkl', 'wb') as f:
#         pickle.dump(dataset, f)

#     transformed_dataset = np.array([output[1] for output in outputs], dtype='float32')
#     print(transformed_dataset)
#     print(transformed_dataset.shape)
#     with open('./transformed_dataset/testset.pkl', 'wb') as f:
#         pickle.dump(transformed_dataset, f)

#     # av2 train -> trainset and valset
#     files = list(Path('./train').rglob('*.parquet'))
#     with tqdm_joblib(desc='Preprocessing', total=200000) as progress_bar:
#         outputs = Parallel(n_jobs=8)(delayed(match_sv)(file) for file in files if should_make_list(file))

#     dataset = np.array([output[0] for output in outputs], dtype='float32')
#     print(dataset)
#     print(dataset.shape)
#     with open('./dataset/trainset_0.pkl', 'wb') as f:
#         pickle.dump(dataset[:100000], f)
#     with open('./dataset/trainset_1.pkl', 'wb') as f:
#         pickle.dump(dataset[100000:], f)

#     transformed_dataset = np.array([output[1] for output in outputs], dtype='float32')
#     print(transformed_dataset)
#     print(transformed_dataset.shape)
#     with open('./transformed_dataset/trainset_0.pkl', 'wb') as f:
#         pickle.dump(transformed_dataset[:100000], f)
#     with open('./transformed_dataset/trainset_1.pkl', 'wb') as f:
#         pickle.dump(transformed_dataset[100000:], f)
