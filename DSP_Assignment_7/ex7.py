import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.linalg import inv

#Write a function that takes three inputs namely an input signalx, a stationery zero-meanprocess, and an odd filter length M, and generates output of the Wiener filter coefficientsh.Hint:  You  may  need  to  use  the  autocorrelation  function  that  you  have  written  in  theAssignment 4, and M = 2N + 1.  Note thathcontains symmetric coefficients
def autocorr(signal, maxlags):
    signal -= np.mean(signal)
    
    lag = []
    corr = []
    for l in range(1, maxlags + 1):
        lag.append(l)
        corr.append(np.sum(signal[l:] * signal[:-l]))
    corr /= (len(signal) * np.var(signal))

    corr = np.hstack([np.flip(corr), [1], corr])
    lag = np.hstack([-1 * np.flip(lag), [0], lag])
    return lag, corr

def weiner_filter_coefficient(signal_x, zero_mean_process, M):
    '''
    Function generates Wiener filter coefficients 
    x: input signal 
    process: stationery zero-mean process
    M: odd filter length
    '''

    vec1 , phi_signal = autocorr(signal_x, len(signal_x) ) #autocorrelation vector of the input signal 
    vec2 , phi_noise = autocorr(zero_mean_process,len(zero_mean_process))#autocorrelation vector of the noise signal

    phi_ss = phi_signal - phi_noise

    #M = 2N+1
    N = int((M-1)/2)

    k_mat = np.arange(-N,N+1)
    k_mat = np.tile(k_mat, (2*N+1,1))
    k_mat_transpose =  np.transpose(k_mat)

    index = abs(k_mat-k_mat_transpose)
    linear_index = index.reshape(-1)
    linear_matrix = phi_signal[linear_index]
    #We need to solve AX = B
    A = linear_matrix.reshape(M,M)

    temp_mat = phi_ss[1:N+1].tolist()
    reverse_temp_mat =  phi_ss[N:0:-1].tolist()
    
    B = np.array(reverse_temp_mat+[phi_ss[0]]+temp_mat)
    B = np.reshape(B,(M,1))

    H = np.linalg.solve(A,B)
    return H

#3.2
# Generate  a  sinusoidal  wave  of  amplitude  equals  to  3  Volts,  with  sample  rate  of  20samples/second of a total duration 50 seconds
fs = 20
A = 3
#t  = 50
t = np.linspace(1, 50, 20*50)

#samples = np.arange(t * fs) / fs
signal_sinosoidal = A*np.sin(t/(2*np.pi))
#signal_sinosoidal = A*np.sin(2 * np.pi*samples) 
plt.plot(t, signal_sinosoidal)

# Generate random noise that is sampled from a standard normal distribution.
noise = np.random.normal(0,1,1000) #n : number of elements in array
plt.plot(t, noise)

# Generate a noisy signal from above, based on the assumption that the noise is additive.
noisy_signal = signal_sinosoidal + noise 
plt.plot(t, noisy_signal)

# Generate the Fourier spectrum (i.e.  magnitude) of the three above (item number 1, 2,3) and plot them against frequency.
signal_fft = np.fft.fft(signal_sinosoidal)
noise_fft = np.fft.fft(noise)
noisy_signal_fft = np.fft.fft(noisy_signal)
N = 1000
freq = np.linspace(0.0, 1.0/(50), N)

fig, ax = plt.subplots(3)

ax[0].plot(freq, signal_fft)


ax[1].plot(freq, noise_fft)


ax[2].plot(freq, noisy_signal_fft)

coeff  = weiner_filter_coefficient(noisy_signal, noise , 101)
coeff

#TO-DO
#Convolution of noisy signal with H to obtain filtered output in time and frequency domain
# Compute  the  Wiener  filter  coefficients  using  the  function  that  you  have  written  insection 3.1 (use M = 101),  and obtain the reconstructed / filtered signal in time aswell  as  in  frequency  domain
# Used In built wiener filter just for reference 
from scipy.signal.signaltools import wiener
filtered_signal = wiener(noisy_signal)

filtered_signal_freq = wiener(noisy_signal_fft)
plt.plot(t, filtered_signal)

plt.plot(freq, filtered_signal_freq)

#  Plot  the  three  results  obtained  in  item  number  6  (i.e.   Wiener  filter  coefficients,  fil-tered signal in time domain, and filtered signal in frequency domain) separately withproperly labelled axes and titles.  The plot in frequency domain should also follow therequirement of item number 5.  How do you interpret your result?3
