import matplotlib.pyplot as plt
import numpy as np
import sounddevice
from scipy.io import wavfile
import pdb  
'''
def acorr(signal, maxlags):
    acorr_vector 
    return lag_vector, acorr_vector
'''

def autocorr2(x, maxlags):
    mean = np.mean(x)
    var = np.var(x)
    x = x - mean
    #corr = np.array([np.sum(x[l:] * x[:-l]) for l in range(1, maxlags + 1)]) / (len(x) * var)
    pdb.set_trace()
    lag = []
    corr = []#np.empty((2 * maxlags + 1))
    count = 0
    for l in range(1, maxlags + 1):
        lag.append(l)
        count +=1 
        #pdb.set_trace()
        corr.append(np.sum(x[l:] * x[:-l]))
    
    #corr = np.array(corr)
    #lag = np.array(lag)
    pdb.set_trace()
    corr /= (len(x) * var)
    corr = np.hstack([np.flip(corr), [1], corr])
    lag = np.hstack([-1 * np.flip(lag), [0], lag])
    return lag, corr

maxlags = 16
fs = 48000

samplerate, dsp_recording = wavfile.read('dsp_recorded.wav')
sounddevice.play(dsp_recording, fs)
  
#pdb.set_trace()

#plt.show()

#plt.plot(lag_manual, corr_manual)
#plt.axhline()
#plt.vlines(lag_manual, 0, corr_manual)
#plt.show()

#plot 1 
# predicted signal overlaid on the original signal
plot1 = plt.subplot(1, 2, 1)
plot1.set_title("Original and overlaid predicted signal")
#plot1.plot(dsp_recording)

# Plot autocorrelation
lag_plt, corr_plt, _, _ = plot1.acorr(dsp_recording, maxlags = 16)
  
# Add labels to autocorrelation plot
'''
plt.title("Autocorrelation of Geeksforgeeks' Users data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
'''  
# Display the autocorrelation plot
lag_manual, corr_manual = autocorr2(dsp_recording, maxlags)

pdb.set_trace()
print(np.linalg.norm(corr_manual - corr_plt), np.linalg.norm(lag_manual - lag_plt))

plt.xlabel('number of samples')
plt.ylabel('Intensity')

#plot 2 prediction error

plot2 = plt.subplot(1, 2, 2)
plot2.set_title("error plot")
#plot2.plot(lag_manual, corr_manual)
plot2.axhline()
plot2.vlines(lag_manual, 0, corr_manual)
#plt.show()
#plot2.plot(error, label="error_signal")

#plt.xlabel('number of samples')
#plt.ylabel('Intensity')

plt.show()