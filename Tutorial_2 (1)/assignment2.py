#1.1

import numpy as np
import pdb
import math

def imgmirror(M, h):
    n, m = M.shape
    w = math.floor(h/2)
    O = np.zeros((n+2*w, m+2*w))
    O[w:-w,w:-w] = M
    O[w:-w,:w] = M[:,w-1::-1] 
    O[w:-w,-w:] = M[:,:-w-1:-1] 
    O[:w,w:-w] = M[w-1::-1,:] 
    O[-w:,w:-w] = M[:-w-1:-1,:] 
    return O

# Test imgmirror function
M = np.array([[1, 2], [3, 4]]) 
h = 5
O = imgmirror(M, h) 

# 1.2
def gaussfilter(I1, K):
    h, _ = K.shape
    n, m = I1.shape
   
    I2 = np.empty_like(I1)
    I1 = imgmirror(I1, h)
    for x in range(n):
        for y in range(m):
            I2[x,y] = (I1[x:x+h, y:y+h] * K).sum() 
    return I2

# 1.3
from matplotlib import pyplot as plt

def gausskern(sigma):
    ax = np.linspace(-2., 2., 5) #5x5 kernel
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

I1 = plt.imread('noisycoke.jpg')

sigmas = [1]#[0.25, 0.5, 1, 2, 4]
denoised_imgs = []

for sig in sigmas:
    K = gausskern(sig)
    I2 = gaussfilter(I1, K)
    denoised_imgs.append(I2)

denoised_imgs = np.hstack(denoised_imgs)
plt.imshow(denoised_imgs, cmap='gray')
plt.savefig('denoisedcoke.jpg')

# 1.4
def gaussfilter_sep(I1, K):
    '''
    h = K.size
    n, m = I1.shape
    
    pdb.set_trace()
    I2x = np.empty([n, 5])
    I2y = np.empty([5, m])
    I1 = imgmirror(I1, h)

    Kx = np.broadcast_to()
    for x in range(n):
        I2x[x,:] = (I1[x:x+h,:] * K)
    for y in range(m):
        I2y[:,y] = I1[:,y:y+h] * K         
    return I2
    '''
    pdb.set_trace()
    n, m = I1.shape
    h = K.size
    gausX = np.zeros((n, m-h+1))
    for i, v in enumerate(K):
        gausX += v * I1[:, i:m-h+i+1]
    gausY = np.zeros((gausX.shape[0]-h+1, gausX.shape[1]))
    for i, v in enumerate(K):
        gausY += v * gausX[i:n-h+i+1]
    pdb.set_trace()
    return gausY

def gausskern_sep(sigma):
    ax = np.linspace(-2., 2., 5) #5 kernel
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel / np.sum(kernel)

sigmas = [1]#[0.25, 0.5, 1, 2, 4]
denoised_imgs = []

for sig in sigmas:
    K = gausskern_sep(sig)
    I2 = gaussfilter_sep(I1, K)
    denoised_imgs.append(I2)

denoised_imgs = np.hstack(denoised_imgs)
plt.imshow(denoised_imgs, cmap='gray')
pdb.set_trace()
plt.savefig('denoisedcoke_sep.jpg')
