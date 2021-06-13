import math
import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice
import pdb

# 1.1 Pre-emphasis
def preemphasis(audiostream, alpha=0.95):
    #audiostream = audiostream - alpha * np.roll(audiostream, -1, axis=0)
    emph_audiostream = np.append(audiostream[0, None], audiostream[1:] - alpha * audiostream[:-1], axis=0)
    return emph_audiostream

point8, sample_rate = sf.read('DSP_Assignment_8/point8.au') # (485100, 8), 44100, 11 sec

'''
#sounddevice.play(point8[0], sample_rate)
plt.figure(figsize=(16, 8))    
plt.plot(point8)
plt.title('Audio signal')
plt.xlabel('Time (s)') #TODO samples or time?
plt.ylabel('Amplitude (s)')
#plt.show()
plt.savefig('audio_signal.jpg')
'''
point8_emph = preemphasis(point8)
'''
plt.figure(figsize=(16, 8))  
plt.plot(point8_emph)
plt.title('Audio signal after pre-emphasis')
plt.xlabel('Time (s)') #TODO samples or time?
plt.ylabel('Amplitude (s)')
#plt.show()
plt.savefig('audio_signal_emph.jpg')
'''

# 1.2 Framing and Windowing
def frame_window(signal, frame_shift, frame_width):
    k = signal.shape[0]

'''
frame_shift = 44100
frame_width = 44100
frame_window(point8_emph, frame_shift, frame_width)
'''

frame_size = 0.025
frame_stride = 0.01

#pdb.set_trace()
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(point8_emph)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length, point8_emph.shape[1]))

pdb.set_trace()
pad_signal = np.append(point8_emph, z, axis=0) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T # [num_frames, frame_length]
frames = pad_signal[indices.astype(np.int32, copy=False)]

frames *= np.repeat(np.hamming(frame_length)[None], point8_emph.shape[1], axis=0).T
#np.repeat(np.hamming(frame_length)[None], 8, axis=0).T.shape