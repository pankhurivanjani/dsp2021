import math
import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from matplotlib import cm
import soundfile as sf
import sounddevice
import pdb

# 1.1 Pre-emphasis
def preemphasis(audiostream, alpha=0.95):
    #audiostream = audiostream - alpha * np.roll(audiostream, -1, axis=0)
    emph_audiostream = np.append(audiostream[0, None], audiostream[1:] - alpha * audiostream[:-1], axis=0)
    return emph_audiostream

point8, sample_rate = sf.read('DSP_Assignment_8/point8.au') # (485100, 8), 44100, 11 sec
#sounddevice.play(point8[0], sample_rate)

plt.figure(figsize=(16, 8))    
plt.plot(point8)
plt.title('Audio signal')
plt.xlabel('Time (s)') #TODO samples or time?
plt.ylabel('Amplitude (s)')
#plt.show()
plt.savefig('DSP_Assignment_8/audio_signal.jpg')

point8_emph = preemphasis(point8)

plt.figure(figsize=(16, 8))  
plt.plot(point8_emph)
plt.title('Audio signal after pre-emphasis')
plt.xlabel('Time (s)') #TODO samples or time?
plt.ylabel('Amplitude (s)')
#plt.show()
plt.savefig('DSP_Assignment_8/audio_signal_emph.jpg')


# 1.2 Framing and Windowing
def frame_window(signal, frame_shift, frame_width):
    frame_width, frame_shift = frame_width * sample_rate, frame_shift * sample_rate  # Convert from seconds to samples
    # frame_width - 1102, frame_shift - 441
    signal_length = len(signal)
    frame_width = int(round(frame_width))
    frame_shift = int(round(frame_shift))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_width)) / frame_shift))  # Make sure that we have at least 1 frame
    # num_frames - 1098
    pad_signal_length = num_frames * frame_shift + frame_width
    z = np.zeros((pad_signal_length - signal_length, signal.shape[1]))

    #pdb.set_trace()
    pad_signal = np.append(signal, z, axis=0) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_width), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_width, 1)).T # [num_frames, frame_width]
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.repeat(np.hamming(frame_width)[None], point8_emph.shape[1], axis=0).T #(1098, 1102, 8)
    return frames

frame_width = 0.025 #25 ms
frame_shift = 0.01 # 10 ms

#pdb.set_trace()
frames = frame_window(point8_emph, frame_shift, frame_width)

# 1.3 Mel-Filterbank
def hz2mel(f_hz):
    f_mel = 2595 * math.log10(1 + f_hz / 700)
    return f_mel

def mel2hz(f_mel):
    f_hz = (700 * (10**(f_mel / 2595) - 1))
    return f_hz

def mel_filterbank(fl, fh, nfft, fs, L):
    mel_fl = hz2mel(fl)
    mel_fh = hz2mel(fh)
    mel_points = np.linspace(mel_fl, mel_fh, L+2)
    hz_points = mel2hz(mel_points)
    f = np.floor((nfft + 1) * hz_points / fs) 
    filter_bank = np.zeros((L, int(np.floor(nfft / 2 + 1))))
    
    for m in range(1, L+1):
        for k in range(int(f[m-1]), int(f[m])):
            filter_bank[m-1, k] = 2 * (k - f[m-1]) / ((f[m+1] - f[m-1]) * (f[m] - f[m-1]))
        for k in range(int(f[m]), int(f[m+1])):
            filter_bank[m-1, k] = 2 * (f[m+1] - k) / ((f[m+1] - f[m-1]) * (f[m+1] - f[m]))
    return filter_bank

fl = 133 # Hz
fh = 6855 # Hz
fs = 16000 # Hz
nfft = 512 #1024
L = 20 # number of filters
num_ceps = 12

filter_bank = mel_filterbank(fl, fh, nfft, fs, L)

fig = plt.figure(figsize=(16, 8))  
ax = fig.gca()
for filter in filter_bank:
    ax.plot(filter) 
plt.title('Triangular filter banks')
plt.xlabel('Frequency') 
plt.ylabel('Filter')
#plt.show()
plt.savefig('DSP_Assignment_8/mel_filterbank.jpg') 

# 1.4 MFCC Implementation
channel_idx = 0
mag_frames = np.absolute(np.fft.rfft(frames[...,channel_idx], nfft))  # Magnitude of the FFT,
# (1098, 1102, 257) if fft over all channels, (1098, 257) if single channel
pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  # Power Spectrum

filter_banks = np.dot(pow_frames, filter_bank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] 
# Keep 2-13, (num_frames, num_ceps)

fig, ax = plt.subplots()
mfcc_data = np.swapaxes(mfcc, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC Spectrogram')
ax.set_xlabel('Time') 
ax.set_ylabel('Frequency')
plt.savefig('DSP_Assignment_8/mfcc_spectrogram.jpg') 

# 1.5 Dynamic features
def dynamic_feature(mfcc):
    #pdb.set_trace()
    (num_frames, num_ceps) = mfcc.shape
    features = np.zeros((num_ceps))
    for m in range(1, num_ceps):
        denom = 0
        for i in range (1, num_frames-1):
            features[m] += i * (mfcc[i-1][m] - mfcc[i+1][m]) 
            denom += i ** 2
        features[m] /= denom
    return features

pdb.set_trace()
features = dynamic_feature(mfcc)
