import numpy as np
import tensorflow as tf

import lal
import lal.antenna
import lalsimulation

from pycbc.waveform import get_td_waveform
#from bbhgen.detector import gen_psd
#from bbhgen.signal import gen_polarisations
#from bbhgen.simulate import ParameterGenerator
#from bbhgen.utils import whiten_data, tukey

import scipy.ndimage as sciim
from scipy.signal import butter, lfilter

def sinegaussian(sample_rate,n_signals):
    phi = np.random.uniform(0,2*np.pi)
    t = np.linspace(0,1,sample_rate)
    A = 1.0
    f_0 = np.random.uniform(70,250,n_signals)
    t_0 =  np.random.uniform(0.4,0.6,n_signals)
    Q = np.random.uniform(3,20,n_signals)
    tau = quality_factor_conversion(Q,f_0)#np.random.uniform(1.0/60.0,1.0/15.0,n_signals)
    f_0 = np.expand_dims(f_0,axis=1)
    t_0 = np.expand_dims(t_0,axis=1)
    tau = np.expand_dims(tau,axis=1)
    h_1 = A * np.exp(-1.0*(t-t_0)**2/(tau**2))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
    h_1 = rescale(h_1)
    return h_1

def ringdown(sample_rate,n_signals):
    t = np.linspace(0,1,sample_rate)
    phi = np.random.uniform(0,2*np.pi)
    A = 1.0
    f_0 = np.random.uniform(70,250,n_signals)
    t_0 = np.random.uniform(0.4,0.6,n_signals)
    Q = np.random.uniform(9,20,n_signals)
    tau = quality_factor_conversion(Q,f_0)
    #tau = np.random.uniform(0.02,0.08,n_signals)
    f_0 = np.expand_dims(f_0,axis=1)
    t_0 = np.expand_dims(t_0,axis=1)
    tau = np.expand_dims(tau,axis=1)
    h_1 = A * np.exp(-1.0*((t-t_0)/(tau)))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
    h_1 = ((t-t_0)>0)*h_1
    h_1 = rescale(h_1)
    return h_1

def whitenoiseburst(sample_rate, n_signals):
    t = np.linspace(0,1,sample_rate)
    t_0 = np.random.uniform(0.4,0.6,n_signals)
    tau = np.random.uniform(1/10000,1/100,n_signals)
    t_0 = np.expand_dims(t_0,axis=1)
    tau = np.expand_dims(tau,axis=1)
    noise = np.array([np.random.normal(0,1,1024) for i in range(n_signals)])
    bandpass = butter_filter(noise, sample_rate, order=5)
    h_1 = np.exp(-(t-t_0)**2/tau) * bandpass
    h_1 = rescale(h_1)
    return h_1

def gaussianblip(sample_rate,n_signals):
    t = np.linspace(0,1,sample_rate)
    t_0 =  np.random.uniform(0.4,0.6,n_signals)
    tau = np.random.uniform(1.0/100.0,1.0/20.0,n_signals)
    t_0 = np.expand_dims(t_0,axis = 1)
    tau = np.expand_dims(tau,axis = 1)
    h_1 = np.exp(-(t-t_0)**2/(tau**2))
    return h_1

def quality_factor_conversion(Q,f_0):
    tau = Q/(np.sqrt(2)*np.pi * f_0)
    return tau

def rescale(x):
    abs_max = np.max(x,axis=1)
    abs_max = np.expand_dims(abs_max, axis=1)
    return 2. * ((x + abs_max) / (2. * abs_max)) - 1.

'''
def bbhinspiral(sample_rate,n_signals):
    T_obs = 1.0

    h_1 = np.zeros([n_signals,sample_rate])
    par_gen = ParameterGenerator(mmin=30., mmax=70., mdist='astro')

    psd_H1 = gen_psd(sample_rate, T_obs, op='Design', det='H1', safe=1.0)
    psd_H1 = psd_H1.data.data

    for i in range(n_signals):
        pars = par_gen.generate_parameters(sample_rate, T_obs,beta=[0.4, 0.6], safe=1.0)
        hp, hc = gen_polarisations(sample_rate, T_obs, par=pars,truncate=False,safe=1.0)

        #whitened_signal = whiten_data(hp, T_obs, sample_rate, psd_H1, flag='td',safe=1.0)
        #normalised = rescale(whitened_signal)
        h_1[i] = hp
    h_1 = rescale(h_1)
    return h_1
'''

def bbhinspiral(sample_rate, n_signals):
    ma1 = np.random.uniform(30, 70, n_signals)
    ma2 = np.random.uniform(30, 70, n_signals)
    h_1 = np.zeros([n_signals,sample_rate])
    
    for i in range(n_signals):
        if ma1[i] >= ma2[i]:
            m1 = ma1[i]
            m2 = ma2[i]
        else:
            m2 = ma1[i]
            m1 = ma2[i]
        hp, hc = get_td_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m2, 
                                f_lower=30, delta_t=1.0/1024, inclination=0.0,
                                distance=100)
        hp = np.array(hp)[-1024:]
        h_1[i] = hp
    h_1 = rescale(h_1)
    return h_1

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    d, c = butter(order, normal_cutoff, btype='high', analog=False)
    return d, c

def butter_filter(data, fs, order=5):
    low_cutoff = 250  # desired cutoff frequency of the filter, Hz
    high_cutoff = 70
    b, a = butter_lowpass(low_cutoff, fs, order=order)
    low_filter = lfilter(b, a, data)
    d, c = butter_highpass(high_cutoff, fs, order=order)
    y = lfilter(d, c, low_filter)
    return y

