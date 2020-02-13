import os
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv1D, Conv2D
from keras.layers import UpSampling1D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from keras.layers.core import Lambda
from generate_training_data import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

model_size = 2
n_classes = 5
T_obs = 1.0
sample_rate = 1024
dt = 1.0/sample_rate
n_signals = 80000

# define the standalone discriminator model
def define_discriminator(in_shape=(1024,2), n_classes=n_classes):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv1D(32*model_size, 7, strides=2, padding='same', kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)
    # normal
    fe = Conv1D(64*model_size, 7, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)
    # downsample to 7x7
    fe = Conv1D(128*model_size, 7, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)
    # normal
    fe = Conv1D(256*model_size, 7, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = SpatialDropout1D(0.5)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model

def define_generator(latent_dim, n_classes=n_classes):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # sky position input
    in_sky_position = Input(shape=(3,))
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 120)(in_label)
    # linear multiplication
    n_nodes = 64*model_size
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((n_nodes, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundatione
    n_nodes = 256 * 64 *model_size
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((64 * model_size, 256))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    merge = Reshape((64,1,514))(merge)
    # upsample
    gen = Conv2DTranspose(256 * model_size, (9,1), strides=(2,1), padding='same', kernel_initializer=init)(merge)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)

    gen = Conv2DTranspose(128 * model_size, (9,1), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(64 * model_size, (9,1), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(32 * model_size, (9,1), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)

    gen = Conv2D(1, (9,1), padding='same', kernel_initializer=init)(gen)
    gen = Activation('tanh')(gen)
    out_layer = Reshape((1024,))(gen)
    # output layer
    out_layer = Concatenate()([out_layer,in_sky_position])
    out_layer = Lambda(TheBox)(out_layer)
    # define model
    model = Model([in_lat, in_label, in_sky_position], [out_layer])
    model.summary()
    return model

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

def TheBox(G_out):
     sample_rate = 1024
     T = 1.0
     df = 1.0/T # hRDCODED FOR 1 SEC LONG OBS

     # split into time series and responses
     x = G_out[:,:sample_rate]
     res = G_out[:,sample_rate:]
     dt = res[:, 0]
     A_H = res[:, 1]
     A_L = res[:, 2]
     pi = tf.constant(np.pi, dtype=tf.complex64)

     # FFT to apply time shift Inverse FFT and apply antenna responses
     x_tilde = tf.signal.rfft(x)
     f = tf.cast(df*np.arange(int(sample_rate/2) + 1),dtype=tf.complex64)
     dt_ex = tf.expand_dims(dt, axis=1)
     dt_ex = tf.cast(dt_ex, dtype=tf.complex64)
     shift = x_tilde * tf.math.exp(2.0*pi*1.0j*dt_ex)
     x_shift = tf.signal.irfft(shift)
     A_H_tensor = tf.expand_dims(tf.cast(A_H, dtype=tf.float32), axis=1)
     A_L_tensor = tf.expand_dims(tf.cast(A_L, dtype=tf.float32), axis=1)
     x = x * A_H_tensor
     x_shift = x_shift * A_L_tensor
     return tf.stack([x,x_shift],axis=-1)

 ##################
 ## data loading ##
 ##################

def load_data(n_signals):
    print('Loading signals...')
    ar = [0] * n_signals
    gen_per_class = n_signals//n_classes
    sg = sinegaussian(sample_rate,gen_per_class)
    rd = ringdown(sample_rate,gen_per_class)
    wn = whitenoiseburst(sample_rate,gen_per_class)
    blip = gaussianblip(sample_rate,gen_per_class)
    bbh = bbhinspiral(sample_rate,gen_per_class)
    trainX = np.concatenate((sg,rd,wn,blip,bbh))
    ar_class = np.arange(0,n_classes)
    classes = np.repeat(ar_class,gen_per_class)
    trainy = ar + classes
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

def generate_latent_points(latent_dim, n_signals, n_classes=n_classes):
    # randn samples from normal distribution
    # side note: the choice of latent space is aribitary as it is meaningless until the network
    ra_rad = np.random.uniform(0,2*np.pi,n_signals)
    de_rad = np.arcsin(np.random.uniform(-1,1,n_signals))
    phase_angle = np.random.uniform(0,np.pi,n_signals)
    res = np.array(response(ra_rad, de_rad, phase_angle,random=False)).T
    # assigns meaning by mapping points from latent space to output
    x_input = randn(latent_dim * n_signals)
    # reshape into a batches
    z_input = x_input.reshape(n_signals, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_signals)
    return [z_input, labels, res]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_signals):
    # generate points in latent space
    z_input, labels_input, res = generate_latent_points(latent_dim, n_signals)
    # predict outputs
    signals = generator.predict([z_input, labels_input, res])
    # create class labels
    y = zeros((n_signals, 1))
    return [signals, labels_input], y

###########
## Plots ##
###########

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dr_losses, label='Discriminitive loss on real')
    plt.plot(df_losses, label='Discriminative loss on fake')
    plt.plot(g_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/data/public_html/2107829/ACGAN/plots/losses.png')
    plt.close()

def save_plot(epoch,g_model,n_samples):
    latent_points, labels, responses = generate_latent_points(latent_dim, n_samples)
    n_class_per_samples = int(n_samples/n_classes)
    # specify labels
    labels = np.asarray([x for _ in range(n_class_per_samples) for x in range(n_classes)])
    # generate images
    X  = g_model.predict([latent_points, labels, responses])

    n_row = 3
    n_col = n_classes
    # plot images

    fig, axs = plt.subplots(n_row, n_col, figsize=(20,10))
    axs = axs.ravel()
    t = np.linspace(0,T_obs,1024)

    for i in range(n_row*n_col):
        axs[i].plot(t,X[i])# plots the signals
    plt.tight_layout()
    plt.savefig("/data/public_html/2107829/ACGAN/plots/generations/generation%s.png" % epoch,dpi=300)
    plt.close()

def save_training_plot(epoch,X):
    n_row = 3
    n_col = n_classes
    # plot images

    fig, axs = plt.subplots(n_row, n_col, figsize=(20,10))
    axs = axs.ravel()
    t = np.linspace(0,T_obs,1024)

    for i in range(n_row*n_col):
        axs[i].plot(t,X[i])# plots the signals
    plt.tight_layout()
    plt.savefig("/data/public_html/2107829/ACGAN/plots/training/training_data%s.png" % epoch,dpi=300)
    plt.close()

###########
## Train ##
##########

g_losses = []
df_losses = []
dr_losses = []

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=128):
     # calculate the number of batches per training epoch
     bat_per_epo = int(n_signals / n_batch)
     # calculate the number of training iterations
     n_steps = bat_per_epo * n_epochs
     # calculate the size of half a batch of samples
     half_batch = int(n_batch / 2)
     # manually enumerate epochs
     for i in range(n_steps):
         # get randomly selected 'real' samples
         [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
         # update discriminator model weights
         _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
         # generate 'fake' examples
         [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
         # update discriminator model weights
         _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
         # prepare points in latent space as input for the generator
         [z_input, z_labels, responses] = generate_latent_points(latent_dim, n_batch)
         # create inverted labels for the fake samples
         y_gan = ones((n_batch, 1))
         # update the generator via the discriminator's error
         _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels, responses], [y_gan, z_labels])
         # summarize loss on this batch
         print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
         g_losses.append(g_1)
         dr_losses.append(d_r1)
         df_losses.append(d_f)
         #d_losses.append(d_loss)
         # evaluate the model performance every 'epoch'
         if (i) % (10000) == 0:
             save_training_plot(i, X_real)
             save_plot(i,g_model,15)
             plot_loss(i)
             g_model.save('/data/public_html/2107829/ACGAN/models/generator%s.h5' % i)
             d_model.save('/data/public_html/2107829/ACGAN/models/discriminator%s.h5' % i)


#########
## Run ##
#########

# size of the latent space
latent_dim = 50
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
#n_classes = 5
#T_obs = 1.0
#dt = 1.0/sample_rate
#sample_rate = 1024
#n_signals = 80000
dataset = load_data(n_signals)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
