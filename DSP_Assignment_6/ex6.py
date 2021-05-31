# 2.1
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

def gen_next_gt(px, py, vx, vy):
    px = px + vx * dt
    py = py + vy * dt 
    return px, py, vx, vy

def gen_next_measurement(px, py, vx, vy, ay, dt):
    px = px + vx * dt
    py = py + vy * dt + 0.5 * ay * (dt)**2
    vy = vy + ay * dt
    return px, py, vx, vy

# 2.2
def add_noise(px, py, vx, vy, mu, var):
    std = math.sqrt(var)
    px += np.random.normal(mu, std)
    py += np.random.normal(mu, std)
    vx += np.random.normal(mu, std)
    vy += np.random.normal(mu, std)
    return px, py, vx, vy 

def get_noise(mu, var):
    std = math.sqrt(var)
    noise = np.diag([np.random.normal(mu, std), np.random.normal(mu, std), np.random.normal(mu, std), np.random.normal(mu, std)])
    return noise

# 2.3
ay = -3.7
dt = 0.01
mu = 0
var_eta = 1.
var_gamma = 0.01
N = 760

px0, py0 = (0, 0)

v = 20 
alpha = 45
vx0 = v * math.cos(math.pi * alpha / 180)
vy0 = v * math.sin(math.pi * alpha / 180)

#pdb.set_trace()
(px, py, vx, vy) = (px0, py0, vx0, vy0)
states_gt = [[px, py, vx, vy]]
for i in range(N):
    px, py, vx, vy = gen_next_gt(px, py, vx, vy)
    px, py, vx, vy = add_noise(px, py, vx, vy, mu, var_gamma)
    states_gt.append([px, py, vx, vy])
states_gt = np.array(states_gt)


(px, py, vx, vy) = (px0, py0, vx0, vy0)
states_measurement = [[px, py, vx, vy]]
for i in range(N):
    px, py, vx, vy = gen_next_measurement(px, py, vx, vy, ay, dt)
    px, py, vx, vy = add_noise(px, py, vx, vy, mu, var_eta)
    states_measurement.append([px, py, vx, vy])
states_measurement = np.array(states_measurement)

(px, py, vx, vy) = (px0, py0, vx0, vy0)
states_optimal = [[px, py, vx, vy]]

A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
b = np.array([0, 0, 0.5 * (dt**2), ay * dt])
std_gamma = math.sqrt(var_gamma)
#gamma = np.diag([np.random.normal(mu, std_gamma), np.random.normal(mu, std_gamma), np.random.normal(mu, std_gamma), np.random.normal(mu, std_gamma)])
gamma = np.identity(4) * var_gamma
C = np.eye(4)
std_eta = math.sqrt(var_eta)
#sigma = np.diag([np.random.normal(mu, std_eta), np.random.normal(mu, std_eta), np.random.normal(mu, std_eta), np.random.normal(mu, std_eta)])
sigma = np.identity(4) * var_eta
V = np.zeros((4, 4))
mu = np.array([px, py, vx, vy])

#TODO merge measurement, optimal and gt to single loop
for i in range(N):
    pdb.set_trace()
    #V = get_noise(mu, var_eta) #TODO what's V?
    P = np.matmul(A, np.matmul(V, A.T)) + gamma #TODO if gamma should change at each iteration?
    t = np.linalg.inv(np.matmul(np.matmul(C, P), C.T) + sigma)
    K = np.matmul(np.matmul(P, C.T), t) 
    px, py, vx, vy = gen_next_measurement(px, py, vx, vy, ay, dt)
    px, py, vx, vy = add_noise(px, py, vx, vy, mu, var_eta)
    x = np.array([px, py, vx, vy])
    tt = x - np.matmul(C, np.matmul(A, mu))
    mu = np.matmul(A, mu) + np.matmul(K, tt)
    states_optimal.append(mu)
    break

states_optimal = np.array(states_optimal)

#states_optimal


plt_px = plt.subplot(2, 2, 1)
#plt.set_title("Original and overlaid predicted signal")
plt_px.plot(states_gt[...,0], label = "Ground truth")
plt_px.plot(states_measurement[...,0], label = "Measurement")
plt.ylabel("X Position (m)")
plt.xlabel("time (sec)")

plt_py = plt.subplot(2, 2, 2)
plt_py.plot(states_gt[...,1], label = "Ground truth")
plt_py.plot(states_measurement[...,1], label = "Measurement")
plt.ylabel("Y Position (m)")
plt.xlabel("time (sec)")

plt_vx = plt.subplot(2, 2, 3)
plt_vx.plot(states_gt[...,2], label = "Ground truth")
plt_vx.plot(states_measurement[...,2], label = "Measurement")
plt.ylabel("X Velocity (m)")
plt.xlabel("time (sec)")

plt_vy = plt.subplot(2, 2, 4)
plt_vy.plot(states_gt[...,3], label = "Ground truth")
plt_vy.plot(states_measurement[...,3], label = "Measurement")
plt.ylabel("Y Velocity (m)")
plt.xlabel("time (sec)")

plt.show()

