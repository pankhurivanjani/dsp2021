# 2.1
 
import matplotlib.pyplot as plt
import numpy as np
import sounddevice
from scipy.io import wavfile
import pdb  

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

maxlags = 16

samplerate, dsp_recording = wavfile.read('dsp_recorded.wav')
#sounddevice.play(dsp_recording, samplerate)

plt_auto = plt.subplot(1, 2, 1)
plt_auto.set_title("Built-in acorr")
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')

lag_auto, corr_auto, _, _ = plt_auto.acorr(dsp_recording, maxlags = 16)
  
plt_manual = plt.subplot(1, 2, 2)
plt_manual.set_title("Custom accor")
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')

lag_manual, corr_manual = autocorr(dsp_recording, maxlags)
plt_manual.axhline()
plt_manual.vlines(lag_manual, 0, corr_manual)

#plt.show()
#print(np.linalg.norm(corr_manual - corr_auto), np.linalg.norm(lag_manual - lag_auto))

# 2.2
from scipy.linalg import solve_toeplitz, toeplitz

def sml_phi(signal, order):
    smlmatrix = np.zeros((order))
    for i in range(1, order+1):   #matrix of size order          
        samples_sum = 0
        for k in range(len(signal)):
            samples_sum += signal[k-i] * signal[k]
        smlmatrix[i-1]+= samples_sum    
    return smlmatrix

def cap_phi(signal, order):
    capmatrix = np.zeros((order,order))
    for i in range(1, order+1): #matrix of size order*order
        for j in range(1, order+1):
            samples_sum = 0
            for k in range(len(signal)):
                samples_sum += signal[k-i] * signal[k-j]
            capmatrix[i-1,j-1]+= samples_sum
    return capmatrix   
'''
def levinson_durbin(corr, order):
    a_coeff = solve_toeplitz((corr[:order], corr[:order]), corr[1:order+1])
    pdb.set_trace()
    e = corr[0]
    a = np.zeros(order+1)
    k = np.zeros(order+1)
    for i in range(1, order+1):
        sum = 0
        for j in range(1, i):
            sum += a[j] * corr[i-j]
        k[i] = (corr[i] - sum) / e
        a[i] = k[i]
        for j in range(1, i):
            a[j] = a[j] - k[i] * a[i-j]
        e = (1 - (k[i]**2)) * e
        pdb.set_trace()
        
    pdb.set_trace()
    print(np.linalg.norm(a - a_coeff))
    return a_coeff
'''
def levinson_durbin(corr, order):
    a_coeff = solve_toeplitz((corr[:order], corr[:order]), corr[1:order+1])
    pdb.set_trace()
    e = corr[0]
    a = np.zeros(order+1)
    k = np.zeros(order+1)
    for i in range(1, order+1):
        sum = 0
        for j in range(1, i):
            #sum += a[j] * corr[i-j]
            sum += corr[1] * a[i-j]
        k[i] = (corr[i] - sum) / e
        a[i] = k[i]
        for j in range(1, i):
            a[j] = a[j] - k[i] * a[i-j]
        e = (1 - (k[i]**2)) * e
        #pdb.set_trace()
        
    pdb.set_trace()
    print(np.linalg.norm(a - a_coeff))
    return a_coeff

def ld(r, p):
    # solve the toeplitz system using the Levinson-Durbin algorithm
    pdb.set_trace()
    g = r[1] / r[0]
    a = np.array([g])
    v = (1. - g * g) * r[0]
    for i in range(1, p):
        g = (r[i+1] - np.dot(a, r[1:i+1])) / v
        a = np.r_[ g,  a - g * a[i-1::-1] ]
        v *= 1. - g*g
    # return the coefficients of the A(z) filter
    return np.r_[1, -a[::-1]]

def solve_coef(signal, order): 
    capmatrix = cap_phi(signal, order)
    smlmatrix = sml_phi(signal, order)
    a = np.linalg.solve(capmatrix, smlmatrix)
    return a  

#a_coeff = levinson_durbin(corr_manual[maxlags:], maxlags)
a_coeff = ld(corr_manual[maxlags:], maxlags)
print(a_coeff)
dsp_lpc = solve_coef(dsp_recording, maxlags) 
print(np.linalg.norm(dsp_lpc + a_coeff[1:]))
