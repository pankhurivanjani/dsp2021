import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import *
def directivity(x, theta, phi):
# def directivity():

  n = len(x)  #number of sensors
  omega = 100
  t = symbols('t')
  T = symbols('T')
  c = 330
  R = 1
    
  s1 = R*cos(theta)*sin(phi)
  s2 = R*sin(theta)*sin(phi)
  s3 = R*cos(phi)

  source = [s1,s2,s3]

  signal = sin(omega*t)
  #power of original signal
  power_signal_single =  (1/(2*T))*integrate(signal*signal,(t, -T,T))
  power_signal_single = limit(power_signal_single,T,oo)

  average_signal_expr = 0
  for i in range(n):
    dk = sqrt(Pow((x[i][0]-source[0]),2) + Pow((x[i][1]-source[1]),2) + Pow((x[i][2]-source[2]),2))
    d0 = sqrt(Pow(x[i][0]-0,2) + Pow(x[i][1]-1,2) + Pow(x[i][2]-1,2))

    average_signal_expr = average_signal_expr + sin(omega * (t - (dk-d0)/c))

  average_signal_expr = average_signal_expr/n

  average_power_integral_expr = (1/(2*T))*integrate(simplify(Pow(average_signal_expr,2)), (t, -T, T))
  average_power = limit(simplify(average_power_integral_expr), T, oo)

  return average_power/power_signal_single
  
x=[[1,0,0],[0,1,0],[0,-1,0],[-1,0,0]]
theta = symbols('theta')
phi = symbols('phi')
intermediate= directivity(x, theta, phi)
#intermediate = simplify(intermediate)

print(intermediate)