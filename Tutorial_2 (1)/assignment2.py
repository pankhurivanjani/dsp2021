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
   
    I2 = np.empty(I1.shape)
    I1 = imgmirror(I1, h)
    for x in range(n):
        for y in range(m):
            I2[x,y] = (I1[x:x+h, y:y+h] * K).sum() 
    return I2

# 1.3
from matplotlib import pyplot as plt

def gausskern(sigma):
    ax = np.linspace(-2., 2., 5) 
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel) # kernel length - 5x5

I1 = plt.imread('noisycoke.jpg')

sigmas = [0.25, 0.5, 1, 2, 4]
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
    n, m = I1.shape
    h = K.size
    I1 = imgmirror(I1, h)

    gausX = np.zeros((n+h-1, m))
    for i, v in enumerate(K):
        gausX += v * I1[:, i:m+i]
    gausY = np.zeros((n, m))
    for i, v in enumerate(K):
        gausY += v * gausX[i:n+i]
    return gausY

def gausskern_sep(sigma):
    ax = np.linspace(-2., 2., 5) 
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma)) # kernel length - 5
    return kernel / np.sum(kernel)

sigmas = [0.25, 0.5, 1, 2, 4]
denoised_imgs_sep = []

for sig in sigmas:
    K = gausskern_sep(sig)
    I2 = gaussfilter_sep(I1, K)
    denoised_imgs_sep.append(I2)

denoised_imgs_sep = np.hstack(denoised_imgs_sep)
print(np.linalg.norm(denoised_imgs-denoised_imgs_sep))
plt.imshow(denoised_imgs_sep, cmap='gray')
plt.savefig('denoisedcoke_sep.jpg')
