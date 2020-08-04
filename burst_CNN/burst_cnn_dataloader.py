#!/usr/env/bin python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import json
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


def main(config):

    path = setup_rundir(config['outdir'])
    print('Saving results to: ', path)
    with open(f'{path}/config.json', 'w') as wf:
        json.dump(config, wf, indent=4)

    print('Using following configuration ', config)

    epochs = config['epochs']
    batch_size = config['batch_size']
    prefetch = config['prefetch']

    print('Setting up dataloaders')

    train_dataset = setup_hdf5_dataset(config['training_data'],
            '/time_series', y_dataset='/labels',
            batch_size=batch_size, prefetch=prefetch)
    val_dataset = setup_hdf5_dataset(config['validation_data'],
            '/time_series', y_dataset='/labels',
            batch_size=batch_size, prefetch=prefetch)

    checkpoint_filepath = os.path.join(path,'best_model_bigger_model.hdf5')
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=config['patience'])

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    activations = {
            'swish': tf.keras.activations.swish,
            'relu': tf.keras.activations.relu,
            'elu': tf.keras.activations.elu
            }
    print('Buidling model')
#create model
    layers_list = []
    layers_list += [
            layers.Conv1D(config['filters'][0],
                kernel_size=config['kernel_size'][0],
                strides=config['strides'][0],
                padding='same',
                activation=activations[config['activation']],
                input_shape=(1024, 2))
            ]
    for i in range(1, config['conv_layers']):
        layers_list += [
                layers.Conv1D(config['filters'][i],
                    kernel_size=config['kernel_size'][i],
                    strides=config['strides'][i],
                    padding='same',
                    activation=activations[config['activation']])
                ]

    layers_list += [layers.Flatten()]
    for i in range(config['dense_layers']):
        layers_list += [
                layers.Dense(config['dense_units'][i],
                    activation=activations[config['activation']]),
                ]
    layers_list += [layers.Dense(1, activation='sigmoid')]

    model = keras.Sequential(layers_list)

    model.summary()


#compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print('Modelled compiled')
#train the model
    print('Fitting model')
    history = model.fit(train_dataset, validation_data=val_dataset,
                        callbacks=[callback,model_checkpoint_callback],epochs=epochs)
    print('Plotting loss')
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

    if not len(sys.argv) > 1:
        raise RuntimeError('No config file provided!')


    with open(sys.argv[1], 'r') as rf:
        config = json.load(rf)
    main(config)
