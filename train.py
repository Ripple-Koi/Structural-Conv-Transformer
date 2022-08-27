import time
import wandb
import numpy as np
import tensorflow as tf
from pathlib import Path

from model.sct import structural_conv_transformer
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
    batch_size=64,
    epochs=2,
    frequency=10,
    num_vehicles=6,
    activation="relu",
    learning_rate=0.0001,
    lr_schedule=True,
    d_model=128,
    num_heads=16,
    dropout_rate=0.4,
    rnn_cell="rnn",
    # TODO LR parameters
    # TODO BatchNorm, LayerNorm
    # TODO CNN num, kernel, pool
    # TODO Layer-wise activation, d_model
    # TODO Dataset
)
wandb.init(config=default_configs, project="my-test-project")
cfg = wandb.config

BATCH_SIZE = cfg.batch_size
EPOCHS = cfg.epochs
FREQUENCY = cfg.frequency
NUM_VEHICLES = cfg.num_vehicles
DATASET = cfg.dataset
MODEL = cfg.model
ACTIVATION = cfg.activation
LR = cfg.learning_rate
LR_SCHEDULE = cfg.lr_schedule
D_MODEL = cfg.d_model
NUM_HEADS = cfg.num_heads
DROPOUT_RATE = cfg.dropout_rate
RNN_CELL = cfg.rnn_cell

checkpoint_path = Path(f"./ckpt/{DATASET}/{MODEL}/{FREQUENCY}Hz/test")
checkpoint_path.mkdir(exist_ok=True, parents=True)
result_path = Path(f"./result/{DATASET}/{MODEL}/{FREQUENCY}Hz/test")
result_path.mkdir(parents=True, exist_ok=True)

datagenerator = DataGenerator(FREQUENCY, NUM_VEHICLES)

predictor = structural_conv_transformer(
    num_layers=1,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=D_MODEL,
    pe_input=5 * FREQUENCY,
    rate=DROPOUT_RATE,
    activation=ACTIVATION,
    num_vehicles=NUM_VEHICLES,
    frequency=FREQUENCY,
    rnn_cell=RNN_CELL,
)

if LR_SCHEDULE:
    learning_rate = CustomSchedule(
        peak_lr=LR,
        end_lr=LR / 5,
        total_steps=EPOCHS * len(datagenerator.graph_train) / BATCH_SIZE,
    )
else:
    learning_rate = LR
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
for epoch in range(EPOCHS):
    datagenerator.shuffle_trainset()
    train_loss.reset_states()
    val_loss.reset_states()
    train_loss_log = []
    val_loss_log = []
    # preds = []
    metrics = []

    start = time.time()

    for step, (inp, tar, graph) in enumerate(
        datagenerator.next_train_batch(BATCH_SIZE)
    ):

        train_step(inp, tar, graph)
        train_loss_log.append(tf.get_static_value(train_loss.result()))

        if (step + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} Step {step + 1} Loss {train_loss.result():.4f}")

    mean_train_loss = np.mean(np.array(train_loss_log))
    mean_train_time = (time.time() - start) / (step + 1)
    print(f"Epoch {epoch + 1} Mean Train Loss {mean_train_loss:.4f}")
    print(f"Mean training time for {step + 1} steps: {mean_train_time:.2f} secs\n")

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

    start_infer = time.time()

    for step, (inp, tar, graph, current_frame, ground_truth) in enumerate(
        datagenerator.next_test_batch(BATCH_SIZE)
    ):  # Infer in batch to avoid OOM

        # TODO timestep index slice should be [9::10] for 1Hz
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

np.savetxt(result_path / "metrics.csv", metrics_table, delimiter=",")
