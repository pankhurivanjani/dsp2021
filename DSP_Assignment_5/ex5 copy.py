import pdb 
import math
import sympy
from mpl_toolkits.mplot3d import Axes3D
#import plotly

from matplotlib import pyplot
#Allowed Python libraries to use: math, numpy, sympy, mpl_toolkits.mplot3d, and plotly
# 2.1 Sensitivity pattern of microphone array

def directivity(x, theta, phi):
    s1 = r * math.cos(theta) * math.sin(phi) 
    s2 = r * math.sin(theta) * math.sin(phi) 
    s3 = r * math.cos(phi) 
    omega = 100

    return power_ratio

def sound_wave(omega, t):
    return math.sin(omega * t)

r = 1
pdb.set_trace()
#theta = 90
#phi = 0
sound_source_location = [0, 1, 1]
sound_velocity = 330
directivity(0, 0, 0)

[[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]]
[0, 0, 1], [0, 0, -1]

#plot.
