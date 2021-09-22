import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical
from GAN_training_data import *
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
sns.set_palette('colorblind')

plt.rcParams.update({'font.size': 24})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

batch_size = 512
loss_dir = '/data/www.astro/2107829/McGAN/test/losses.png'
gen_dir = "/data/www.astro/2107829/McGAN/test/generations/generation%s.png"
train_dir = "/data/www.astro/2107829/McGAN/test/training/training%s.png"
gen_model_dir = "/data/www.astro/2107829/McGAN/test/models/cgan_generator%s.h5"

############
## Models ##
############

def define_discriminator(in_shape=1024,n_classes=5):
    # i'm using functional API since its more flexible and can handle mulitple inputs better (not a sequential model).
    # label input
    in_label = Input(shape=(5,))
    # scale up ti image dim with linear activation (linear also means no activation in this case)
    n_nodes = in_shape
    li = Dense(n_nodes)(in_label)
    # reshape to additional channel
    li = Reshape((in_shape,1))(li)
    # image input
    in_image = Input(shape=(1024,))
    In = Reshape((in_shape,1))(in_image)
    # concat label as a channel!
    merge = Concatenate()([In, li])
    # downsample
    fe = Conv1D(64, 14, strides=2, padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)

    fe = Conv1D(128, 14, strides=2, padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)

    fe = Conv1D(256, 14, strides=2, padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)

    fe = Conv1D(512, 14, strides=2, padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)

    # flatten feature map
    fe = Flatten()(fe)
    # Dropout
    #fe = Dropout(0.5)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # complie that mofo
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def define_generator(latent_dim, n_classes=5):
    # image input
    in_lat = Input(shape=(latent_dim,))
    in_label = Input(shape=(n_classes,))

    x = Concatenate()([in_lat, in_label])

    n_nodes = 64 * 512
    merge = Dense(n_nodes)(x)
    merge = Activation('relu')(merge)

    # Add dimension as there's no 1DTranspose
    merge = Reshape((64,1,512))(merge)
    # upsample
    #gen = Conv2DTranspose(512, kernel_size=(18,1), strides=(1,1), padding='same')(merge)
    #gen = Activation('relu')(gen)
    #gen = BatchNormalization()(gen)

    gen = Conv2DTranspose(256, kernel_size=(18,1), strides=(2,1), padding='same')(merge)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(128, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(64, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(1, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = Activation('linear')(gen)
   # output
    out_layer = Reshape((1024,))(gen)
    # define model
    model = Model([in_lat, in_label], [out_layer])
    model.summary()
    return model

# the combined generator and discirminator, used to update the generator
def define_gan(g_model, d_model):
    # make weights in D not trainable
    d_model.trainable = False
    # get noise and label inputs from G
    gen_noise, gen_label  = g_model.input
    # get image output from G
    gen_output = g_model.output
    # connect image output and label input from G as inputs to D
    gan_output = d_model([gen_output,gen_label])
    # let the gan model take noise and label then output classification
    model = Model([gen_noise,gen_label], gan_output)
    # compile
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

##################
## data loading ##
##################

def load_data(n_signals):
    ar = [0] * n_signals
    gen_per_class = n_signals//n_classes
    sg = sinegaussian(sample_rate,gen_per_class)
    rd = ringdown(sample_rate,gen_per_class)
    wn = whitenoiseburst(sample_rate,gen_per_class)
    blip = gaussianblip(sample_rate,gen_per_class)
    bbh = bbhinspiral(sample_rate,gen_per_class)
    trainX = np.concatenate((sg,rd,wn,blip,bbh),axis=0)
    ar_class = np.arange(0,n_classes)
    classes = np.repeat(ar_class,gen_per_class)
    trainy = ar + classes
    trainy = to_categorical(trainy)
    return [trainX, trainy]

def generate_real_samples(dataset, n_signals):
    # split into signals and labels
    signals, labels = dataset
    # choose randomly
    idx = randint(0, signals.shape[0], n_signals)
    # select images and labels
    X, labels = signals[idx], labels[idx]
    # generate class labels
    y = np.ones((n_signals, 1))
    return [X,labels], y

#########################
## required randomness ##
#########################

def generate_latent_points(latent_dim, n_signals, n_classes=5):
    # randn samples from normal distribution
    # side note: the choice of latent space is aribitary as it is meaningless until the network
    # assigns meaning by mapping points from latent space to output
    x_input = randn(latent_dim * n_signals)
    # reshape into a batches
    z_input = x_input.reshape(n_signals, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_signals)
    labels = to_categorical(labels)
    return [z_input, labels]

# use generator to make fake examples with classes
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict
    images = generator.predict([z_input, labels_input])
    # generate class labels
    y = zeros((n_samples,1))
    return [images,labels_input], y

###########
## Plots ##
###########

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_loss(epoch,):
    fig, ax1 = plt.subplots(figsize=(12,8))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(d_losses, 'C0', linestyle='-', label='Discriminitive loss')
    ax1.plot(g_losses, 'C1', linestyle='--', label='Generative loss')
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylim(0,1)
    ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
    ax2.plot(accuracy1, 'C2', linestyle='-.', label='Accuracy on real')
    ax2.plot(accuracy2,'C4', linestyle=':', label='Accuracy on fake')
    ax2.grid()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=2)
    fig.savefig(loss_dir, bbox_inches='tight', dpi=300)
    plt.close()

def save_gen_plot(epoch,g_model, n_row, n_col):
    # plot images
    latent_points, _ = generate_latent_points(100, 20)
    # specify labels
    labels = np.asarray([x for _ in range(4) for x in range(5)])
    labels = to_categorical(labels)
    # generate images
    X  = g_model.predict([latent_points, labels])


    fig, axs = plt.subplots(n_row,n_col, figsize=(20,10))
    #plt.subplots_adjust(wspace=0.03, hspace=0.05)
    axs = axs.ravel()


    t = np.linspace(0,1,1024)

    for i in range(n_col * n_row):
        axs[i].plot(t,X[i],'#ee0000', linewidth=0.5)
        axs[i].grid()
        axs[i].set_xlim(0.3,0.7)
    #plt.tight_layout()
    plt.savefig(gen_dir % epoch,dpi=300)
    plt.close()

def save_training_plot(epoch,X):
    n_row = 4
    n_col = 5
    # plot images

    fig, axs = plt.subplots(n_row, n_col, figsize=(20,10))
    axs = axs.ravel()
    t = np.linspace(0,1,1024)

    for i in range(n_row*n_col):
        axs[i].plot(t,X[i],'#ee0000', linewidth = 0.5)# plots the signals
        axs[i].set_xlim(0.3,0.7)
        axs[i].grid('on')
    plt.tight_layout()
    plt.savefig(train_dir % epoch,dpi=300)
    plt.close()

###########
## train ##
###########

g_losses = []
d_losses = []
accuracy1 = []
accuracy2 = []

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=512):
    # take a batch from the data
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # half for training half for checking
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        running_g = []
        running_d = []
        running_a1 = []
        running_a2 = []
        # enumerate batches over the trainig set, probably wont need this loop if generating on the go
        for j in range(bat_per_epo):
            # get real images
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator weights
            d_loss1, acc1 = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate fakes, requires latent dim to produce input for generator
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator weights
            d_loss2, acc2 = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as inputs for generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fakes, this is how G tries to trick D
            y_gan = ones((n_batch,1))
            # update the generator via discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            print('>%d, %d%d, d1=%.3f, d2=%.3f g=%.3f, a1=%.3f, a2=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, acc1, acc2))
            running_g.append(g_loss)
            running_d.append(d_loss2)
            running_a1.append(acc1)
            running_a2.append(acc2)

        g_losses.append(np.mean(running_g))
        d_losses.append(np.mean(running_d))
        accuracy1.append(np.mean(running_a1))
        accuracy2.append(np.mean(running_a2))
        save_gen_plot(i,g_model,4,5)
        save_training_plot(i,X_real)
        plot_loss(i)
        g_model.save(gen_model_dir % i)

#########
## Run ##
#########

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
n_classes = 5
n_timesteps = 1024
sample_rate = 1024
n_signals = 100000
dataset = load_data(n_signals)
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_batch=batch_size)
