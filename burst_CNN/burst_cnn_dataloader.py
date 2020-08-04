#!/usr/env/bin python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(4)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def setup_rundir(path, make_dir=True):
    run = 0
    rundir = path + f"run_{run}/"
    while os.path.exists(rundir):
        run += 1
        rundir = path + f"run_{run}/"
    if make_dir:
        os.makedirs(rundir, exist_ok=False)
    return rundir


def setup_hdf5_dataset(filename,  x_dataset, y_dataset=None,
                       batch_size=100, prefetch=1):
    """
    Setup a Tensorflow dataset from a HDF5 file use Tensorflow io.

    Allows for batch size and prefectching.

    NOTE: dataset names much contain the complete path, e.g.: '/x_data'
    """
    x_data = tfio.IODataset.from_hdf5(filename, dataset=x_dataset)
    x_data = x_data.map(lambda x : tf.reshape(x, (1024, 2)))
    if y_dataset is not None:
        y_data = tfio.IODataset.from_hdf5(filename, dataset=y_dataset)
        return tf.data.Dataset.zip((x_data, y_data)).batch(batch_size, drop_remainder=False).prefetch(prefetch)
    else:
        return x_train.batch(batch_size, drop_remainder=False).prefetch(prefetch)


def swish(x):
    return tf.keras.activations.swish(x)


def main(path, filename, val_filename):

    path = setup_rundir(path)
    print('Saving results to: ', path)

    epochs = 5
    batch_size = 1000
    prefetch = 10

    train_dataset = setup_hdf5_dataset(filename, '/time_series', y_dataset='/labels',
            batch_size=batch_size, prefetch=prefetch)
    val_dataset = setup_hdf5_dataset(val_filename, '/time_series', y_dataset='/labels',
            batch_size=batch_size, prefetch=prefetch)

    checkpoint_filepath = os.path.join(path,'best_model_bigger_model.hdf5')
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

#create model
    model = keras.Sequential(
        [
            layers.Conv1D(16, kernel_size=5, strides=2, padding='same', activation=swish, input_shape=(1024, 2)),
            layers.Conv1D(16, kernel_size=5, strides=2, padding='same', activation=swish),
            layers.Conv1D(8, kernel_size=5, strides=2, padding='same', activation=swish),
            layers.Conv1D(8, kernel_size=5, strides=2, padding='same', activation=swish),
            layers.Flatten(),
            layers.Dense(100),
            layers.Dense(1, activation='sigmoid'),
        ]
    )
    model.summary()


#compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#train the model
    history = model.fit(train_dataset, validation_data=val_dataset,
                        callbacks=[callback,model_checkpoint_callback],epochs=epochs)

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(history.history['loss'], label = "train_loss")
    axs[0].plot(history.history['val_loss'], label = "val_loss")

    axs[1].plot(history.history['accuracy'], label = "train_acc")
    axs[1].plot(history.history['val_accuracy'], label = "val_acc")

    axs[0].set_title("Training Loss")
    axs[1].set_title("Training Accuracy")

    axs[0].set_xlabel("Epoch #")
    axs[1].set_xlabel("Epoch #")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")
    axs[0].legend()
    axs[1].legend()
    plt.savefig(os.path.join(path,'loss_acc_bigger_network.png'))

if __name__ == '__main__':

    path = './outdir/generations_snr_1_16/'
    filename = 'data/train_G_200k.h5'
    val_filename = 'data/train_G.h5'

    main(path, filename, val_filename)
