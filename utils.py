import tensorflow as tf
import numpy as np
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, FREQUENCY, NUM_VEHICLES):
        self.frequency, self.num_vehicles = FREQUENCY, NUM_VEHICLES
        # load data here
        data_dir = "./data/val/val.npz"
        graph = np.stack(
            [
                np.load(data_dir)["GLOBAL"][:, :: int(10 / FREQUENCY), :, :],
                np.load(data_dir)["MEDIUM"][:, :: int(10 / FREQUENCY), :, :],
                np.load(data_dir)["LOCAL"][:, :: int(10 / FREQUENCY), :, :],
            ],
            axis=1,
        )
        feature_index = []
        for i in range(NUM_VEHICLES):
            feature_index.append(3 + i * 5)
            feature_index.append(4 + i * 5)
        traj = np.load(data_dir)["TRANSFORM"][:, :, feature_index].astype("float32")
        self.graph_train, self.graph_test, traj_train, traj_test = train_test_split(
            graph, traj, train_size=19200, random_state=42
        )
        self.input_traj_train, self.target_traj_train = juggle_and_split(
            traj_train, NUM_VEHICLES
        )
        self.input_traj_test, self.target_traj_test = juggle_and_split(
            traj_test, NUM_VEHICLES
        )

    def shuffle_trainset(self):
        self.graph_train, self.input_traj_train, self.target_traj_train = shuffle(
            self.graph_train, self.input_traj_train, self.target_traj_train
        )

    def next_train_batch(self, batch_size):
        train_steps = len(self.graph_train) // batch_size
        for step in range(train_steps):
            yield (
                self.input_traj_train[
                    step * batch_size : (step + 1) * batch_size,
                    int(10 / self.frequency) - 1 :: int(10 / self.frequency),
                    :,
                ],
                self.target_traj_train[
                    step * batch_size : (step + 1) * batch_size,
                    int(10 / self.frequency)
                    - 1 :: int(10 / self.frequency) * self.num_vehicles,
                    :,
                ],
                self.graph_train[
                    step * batch_size : (step + 1) * batch_size, :, :, :, :
                ],
            )

    def next_test_batch(self, batch_size):
        test_steps = len(self.graph_test) // batch_size
        for step in range(test_steps):
            yield (
                self.input_traj_test[
                    step * batch_size : (step + 1) * batch_size,
                    int(10 / self.frequency) - 1 :: int(10 / self.frequency),
                    :,
                ],
                self.target_traj_test[
                    step * batch_size : (step + 1) * batch_size,
                    int(10 / self.frequency)
                    - 1 :: int(10 / self.frequency) * self.num_vehicles,
                    :,
                ],
                self.graph_test[
                    step * batch_size : (step + 1) * batch_size, :, :, :, :
                ],
                self.input_traj_test[
                    step * batch_size : (step + 1) * batch_size,
                    -self.num_vehicles : -self.num_vehicles + 1,
                    :,
                ],
                self.target_traj_test[
                    step * batch_size : (step + 1) * batch_size, :: self.num_vehicles, :
                ],
            )


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, peak_lr=0.001, end_lr=0.0001, total_steps=3000
    ):  # total training steps around 70000
        super(CustomSchedule, self).__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = tf.math.square(end_lr / peak_lr) * total_steps

    def __call__(self, step):
        warmup = self.peak_lr / self.warmup_steps * step
        decrease = self.peak_lr * tf.math.sqrt(self.warmup_steps) * tf.math.rsqrt(step)

        return tf.math.minimum(warmup, decrease)


# TODO: weighted MAE
def loss_function(real, pred):
    loss = tf.math.reduce_mean(abs(real - pred))
    return loss


def juggle_and_split(data, num_vehicles):
    # Juggle the dimensions to (batch, frame * vehicle, feature)
    data = np.reshape(data, (-1, 110, num_vehicles, 2))
    data = np.reshape(data, (-1, 110 * num_vehicles, 2))
    # split 5 for past, 6 for future
    input_tensor = data[:, : 50 * num_vehicles, :]
    target_tensor = data[:, -60 * num_vehicles :, :]
    return input_tensor, target_tensor


def interpolate_pred(pred, current):
    x = np.linspace(0, 6, num=1 + len(pred[0, :, 0]))
    y = np.concatenate([current, pred], axis=1)
    xnew = np.linspace(0, 6, num=61)
    f = interpolate.interp1d(x, y, kind="cubic", axis=1)
    interp_pred = f(xnew)[:, 1:, :]
    return interp_pred


def v_sum(vector):
    return np.sqrt(np.square(vector[:, :, 0]) + np.square(vector[:, :, 1]))


def evaluation(interp_pred, groundtruth):
    """Calculate mean and final longitudinal and later position error, fde and ade.

    Args:
        interp_pred (NDArray): _description_
        groundtruth (NDArray): _description_

    Returns:
        NDArray: _description_
    """
    delta_v = interp_pred - groundtruth
    delta_d = delta_v.cumsum(axis=1) * 0.1  # [N, T, 2]

    final_long_error = abs(delta_d[:, 9::10, 0])  # [N, 6]
    final_lat_error = abs(delta_d[:, 9::10, 1])  # [N, 6]
    final_distance_error = v_sum(delta_d[:, 9::10, :])  # [N, 6]

    average_long_error = np.stack(
        [
            np.mean(abs(delta_d[:, 0:10, 0]), axis=1),
            np.mean(abs(delta_d[:, 0:20, 0]), axis=1),
            np.mean(abs(delta_d[:, 0:30, 0]), axis=1),
            np.mean(abs(delta_d[:, 0:40, 0]), axis=1),
            np.mean(abs(delta_d[:, 0:50, 0]), axis=1),
            np.mean(abs(delta_d[:, 0:60, 0]), axis=1),
        ],
        axis=1,
    )  # [N, 6]
    average_lat_error = np.stack(
        [
            np.mean(abs(delta_d[:, 0:10, 1]), axis=1),
            np.mean(abs(delta_d[:, 0:20, 1]), axis=1),
            np.mean(abs(delta_d[:, 0:30, 1]), axis=1),
            np.mean(abs(delta_d[:, 0:40, 1]), axis=1),
            np.mean(abs(delta_d[:, 0:50, 1]), axis=1),
            np.mean(abs(delta_d[:, 0:60, 1]), axis=1),
        ],
        axis=1,
    )  # [N, 6]
    average_distance_error = np.stack(
        [
            np.mean(v_sum(delta_d[:, 0:10, :]), axis=1),
            np.mean(v_sum(delta_d[:, 0:20, :]), axis=1),
            np.mean(v_sum(delta_d[:, 0:30, :]), axis=1),
            np.mean(v_sum(delta_d[:, 0:40, :]), axis=1),
            np.mean(v_sum(delta_d[:, 0:50, :]), axis=1),
            np.mean(v_sum(delta_d[:, 0:60, :]), axis=1),
        ],
        axis=1,
    )  # [N, 6]
    return np.stack(
        [
            np.mean(final_long_error, axis=0),
            np.mean(final_lat_error, axis=0),
            np.mean(final_distance_error, axis=0),
            np.mean(average_long_error, axis=0),
            np.mean(average_lat_error, axis=0),
            np.mean(average_distance_error, axis=0),
        ],
        axis=0,
    )  # [6, 6]
