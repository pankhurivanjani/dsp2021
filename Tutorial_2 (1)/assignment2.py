#1.1

import numpy as np
import pdb

def imgmirror(M, w):
    n, m = M.shape
    O = np.zeros((n+2*w, m+2*w))
    O[w:-w,w:-w] = M
    O[w:-w,:w] = M[:,w-1::-1] 
    O[w:-w,-w:] = M[:,:-w-1:-1] 
    O[:w,w:-w] = M[w-1::-1,:] 
    O[-w:,w:-w] = M[:-w-1:-1,:] 
    '''
    O[w:-w,:w]  = M[:,:w]
    O[w:-w,-w:] = M[:,-w:]
    O[:w,w:-w]  = M[:w,:]
    O[-w:,w:-w] = M[-w:,:]
    '''
    return O

M = np.array([[ 1, 2], [3, 4]]) # (2,2)
w = 2
O = imgmirror(M, w) # (4,4)
# 1.2
import math

def gaussfilter(I1, K):
    h, _ = K.shape
    n, m = I1.shape
    a = math.ceil(h/2)
   
    I2 = np.empty_like(I1)
    I1 = imgmirror(I1, h//2)
    pdb.set_trace()
    for x in range(n):
        for y in range(m):
            #pdb.set_trace()
            I2[x,y] = (I1[x:x+h, y:y+h] * K).sum() #,y-a+1:y+a-1
            #I2[x,y] += I1[x+i-a,y+j-a] * K[i,j]
    pdb.set_trace()
    return I2

# 1.3
#from PIL import Image
from matplotlib import pyplot as plt

def gausskern(sig):
    """
    creates gaussian kernel with a sigma of sig
    """
    ax = np.linspace(-2., 2., 5)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

#I1 = Image.open('noisycoke.jpg')
I1 = plt.imread('noisycoke.jpg')
K = gausskern(1)
I2 = gaussfilter(I1, K)
plt.imshow(I2, cmap='gray')


# 1.4

def gaussfilter_sep():
    pass    