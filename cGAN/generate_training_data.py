import numpy as np
from lal_response_functions import *
import tensorflow as tf

import lal
import lalsimulation
from math import sqrt

from bbhgen.detector import gen_psd
from bbhgen.signal import gen_polarisations
from bbhgen.simulate import ParameterGenerator
from bbhgen.utils import whiten_data

from sklearn.preprocessing import MinMaxScaler

def sinegaussian(sample_rate,n_signals):
    phi = 2*np.pi
    t = np.linspace(0,1,sample_rate)
    A = 1.0
    f_0 = np.random.uniform(30,50,n_signals)
    t_0 =  np.random.uniform(0.2,0.8,n_signals)
    tau = np.random.uniform(1.0/60.0,1.0/15.0,n_signals)
    f_0 = np.expand_dims(f_0,axis=1)
    t_0 = np.expand_dims(t_0,axis=1)
    tau = np.expand_dims(tau,axis=1)
    h_1 = A * np.exp(-1.0*(t-t_0)**2/(tau**2))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
    res = [response(ra_rad=None,de_rad=None,phase_angle=None,random=True) for _ in range(n_signals)]
    X = np.concatenate([h_1,res],axis=1)
    X = numpy_box(X)
    return X

def ringdown(sample_rate,n_signals):
    t = np.linspace(0,1,sample_rate)
    phi = 2*np.pi
    A = 1.0
    f_0 = np.random.uniform(30,50,n_signals)
    t_0 = np.random.uniform(0.1,0.6,n_signals)
    tau = np.random.uniform(0.02,0.08,n_signals)
    f_0 = np.expand_dims(f_0,axis=1)
    t_0 = np.expand_dims(t_0,axis=1)
    tau = np.expand_dims(tau,axis=1)
    h_1 = A * np.exp(-1.0*((t-t_0)/(tau)))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
    h_1 = ((t-t_0)>0)*h_1         # shifts signal by  making the start time non zero
    res = [response(ra_rad=None,de_rad=None,phase_angle=None,random=True) for _ in range(n_signals)]
    X = np.concatenate([h_1,res],axis=1)
    X = numpy_box(X)
    return X

def whitenoiseburst(sample_rate, n_signals):

    zeros = np.zeros([n_signals,sample_rate])

    for i in range(n_signals):
        size = np.random.randint(100,400)
        noise = np.random.normal(0,0.25,size)
        start = np.random.randint(50,550)
        end = start + size
        idx = np.arange(start,end)
        np.put(zeros[i],idx,noise, mode='clip')

    res = [response(ra_rad=None,de_rad=None,phase_angle=None,random=True) for _ in range(n_signals)] #np.tile((0,1,1),[10,1])
    X = np.concatenate([zeros,res],axis=1)
    X = numpy_box(X)
    return X

def gaussianblip(sample_rate,n_signals):
    t = np.linspace(0,1,sample_rate)
    t_0 =  np.random.uniform(0.2,0.8,n_signals)
    tau = np.random.uniform(1.0/100.0,1.0/20.0,n_signals)
    t_0 = np.expand_dims(t_0,axis = 1)
    tau = np.expand_dims(tau,axis = 1)
    h_1 = np.exp(-(t-t_0)**2/(tau**2))
    res = [response(ra_rad=None,de_rad=None,phase_angle=None,random=True) for _ in range(n_signals)]
    X = np.concatenate([h_1,res],axis=1)
    print(X.shape)
    X = numpy_box(X)
    return X

def rescale(x):
    abs_max = np.max(np.abs(x))#np.max([np.abs(xmax), np.abs(xmin)])
    return 2. * ((x + abs_max) / (2. * abs_max)) - 1.


def bbhinspiral(sample_rate,n_signals):
    T_obs = 1.0

    h_1 = np.zeros([n_signals,sample_rate])
    par_gen = ParameterGenerator(mmin=5., mmax=100., mdist='astro')

    psd_H1 = gen_psd(sample_rate, T_obs, op='AdvDesign', det='H1')
    psd_H1 = psd_H1.data.data

    for i in range(n_signals):
        pars = par_gen.generate_parameters(sample_rate, T_obs)
        hp, hc = gen_polarisations(sample_rate, T_obs, par=pars)

        whitened_signal = whiten_data(hp, T_obs, sample_rate, psd_H1, flag='td')
        normalised = rescale(whitened_signal)
        h_1[i] = normalised

    res = [response(ra_rad=None,de_rad=None,phase_angle=None,random=True) for _ in range(n_signals)]
    X = np.concatenate([h_1,res],axis=1)
    print(X.shape)
    X = numpy_box(X)
    return X

def response(ra_rad,de_rad,phase_angle,random):
    if random:
        ra_rad = np.random.uniform(0,2*np.pi)
        de_rad = np.arcsin(np.random.uniform(-1,1))
        phase_angle = np.random.uniform(0,np.pi)

        dt = H1.time_delay_from_detector(L1,ra_rad,de_rad,1215265166.000)
        A_H = H1.antenna_pattern(ra_rad, de_rad, phase_angle, 1215265166.000)[0]
        A_L = L1.antenna_pattern(ra_rad, de_rad, phase_angle, 1215265166.000)[0]

    else:

        dt = H1.time_delay_from_detector(L1,ra_rad,de_rad,1215265166.000)
        A_H = H1.antenna_pattern(ra_rad, de_rad, phase_angle, 1215265166.000)[0]
        A_L = L1.antenna_pattern(ra_rad, de_rad, phase_angle, 1215265166.000)[0]
    return dt, A_H, A_L

def numpy_box(G_out):
    n = 1024
    T = 1.0
    df = 1.0/T # hRDCODED FOR 1 SEC LONG OBS

    # split into time series and responses
    x = G_out[:,:n]
    res = G_out[:,n:]
    dt = res[:, 0]
    A_H = res[:, 1]
    A_L = res[:, 2]

    # FFT to apply time shift Inverse FFT and apply antenna responses
    x_tilde = np.fft.rfft(x)
    f = df*np.arange(int(n/2) + 1)
    f.astype(complex)
    dt_ex = np.expand_dims(dt, axis=1)
    dt_ex.astype(complex)
    #dt_ex = tf.cast(dt_ex, dtype=tf.complex64)
    shift = x_tilde * np.exp(2.0*np.pi*1.0j*dt_ex)
    x_shift = np.fft.irfft(shift)
    A_H_array = np.expand_dims(A_H, axis=1)
    A_L_array = np.expand_dims(A_L, axis=1)
    x = x * A_H_array
    x_shift = x_shift * A_L_array
    return np.stack([x,x_shift],axis=-1)

H1 = Detector('H1')
L1 = Detector('L1')
