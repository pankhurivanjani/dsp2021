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

plt.show()
#print(np.linalg.norm(corr_manual - corr_auto), np.linalg.norm(lag_manual - lag_auto))

# 2.2
def levinson_durbin_builtin(r, p):
    from scipy.linalg import solve_toeplitz
    a = solve_toeplitz((r[:p], r[:p]), r[1:p+1])
    return a

def levinson_durbin(r, p):
    k = r[1] / r[0]
    a = np.array([k])
    e = (1. - k * k) * r[0]
    for i in range(1, p):
        k = (r[i+1] - np.dot(a, r[1:i+1])) / e
        a = np.r_[k,  a - k * a[i-1::-1]]
        e *= 1. - k*k
    return a[::-1]

a_coeff = levinson_durbin(corr_manual[maxlags:], maxlags)
print(a_coeff)
#a_coeff_builtin = levinson_durbin_builtin(corr_manual[maxlags:], maxlags)
#print(np.linalg.norm(a_coeff_builtin - a_coeff))