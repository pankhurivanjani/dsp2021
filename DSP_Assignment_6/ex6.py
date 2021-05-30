import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

def gen_next_pos_vel(px, py, vx, vy, ay, dt):
    px = px + vx * dt
    py = py + vy * dt + 0.5 * ay * (dt)**2
    vy = vy + ay * dt
    return px, py, vx, vy

def add_noise(px, py, vx, vy, mu, var_eta, var_gamma):
    px += np.random.normal(mu, var_eta)
    py += np.random.normal(mu, var_eta)
    vx += np.random.normal(mu, var_gamma)
    vy += np.random.normal(mu, var_gamma)
    return px, py, vx, vy 

ay = -3.7
dt = 0.01
mu = 0
var_eta = 1.
var_gamma = 0.01
N = 760

px, py = (0, 0)

v = 20 
alpha = 45
vx = v * math.cos(math.pi * alpha / 180)
vy = v * math.sin(math.pi * alpha / 180)

#pdb.set_trace()
gt_model = [[px, py, vx, vy]]
for i in range(N):
    px, py, vx, vy = gen_next_pos_vel(px, py, vx, vy, ay, dt)
    gt_model.append([px, py, vx, vy])

gt_model = np.array(gt_model)

plt_px = plt.subplot(2, 2, 1)
#plt.set_title("Original and overlaid predicted signal")
plt_px.plot(gt_model[...,0], label = "Ground truth")
plt.ylabel("X Position (m)")
plt.xlabel("time (sec)")

plt_py = plt.subplot(2, 2, 2)
plt_py.plot(gt_model[...,1], label = "Ground truth")
plt.ylabel("Y Position (m)")
plt.xlabel("time (sec)")

plt_vx = plt.subplot(2, 2, 3)
plt_vx.plot(gt_model[...,2], label = "Ground truth")
plt.ylabel("X Velocity (m)")
plt.xlabel("time (sec)")

plt_vy = plt.subplot(2, 2, 4)
plt_vy.plot(gt_model[...,3], label = "Ground truth")
plt.ylabel("Y Velocity (m)")
plt.xlabel("time (sec)")

plt.show()