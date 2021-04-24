import numpy as np
import pdb
from PIL import Image

vec1 = np.array([0, 127.5, 127.5])
mat1 = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

def rgb2ycbcr(rgb):
    ycbcr = vec1 + rgb.dot(mat1.T) 
    return ycbcr

def ycbcr2rgb(ycbcr):
    mat1_inv = np.linalg.inv(mat1)
    rgb = (ycbcr - vec1).dot(mat1_inv.T)
    return rgb

img = Image.open("birds.ppm")
#img.show()

img_rgb = np.asarray(img) #default image mode is rgb
img_ycbcr = rgb2ycbcr(img_rgb)

img_ycbcr_view = img_ycbcr.astype('uint8')
img_ycbcr_view = Image.fromarray(img_ycbcr_view, mode='YCbCr')
#img_ycbcr_view.show()

from Utility import blockproc
import math 

def downsample(img, w):
    p, q = img.shape[0]//w, img.shape[1]//w #math.ceil(x/w), math.ceil(y/w)
    out = np.zeros((p, q))
    for i in range(0, p):
        for j in range(0, q):
            out[i, j] = np.mean(img[i * w: (i + 1) * w, j * w: (j + 1) * w])
    return out

def upsample(img, w): #TODO logic?
    out = img.repeat(w, axis=0).repeat(w, axis=1)
    return out

print(img_ycbcr.shape)
w = 8
img_y = downsample(img_ycbcr[...,0], w)
img_cb = downsample(img_ycbcr[...,1], w)
img_cr = downsample(img_ycbcr[...,2], w)

pdb.set_trace()
img_ycbcr = np.stack((img_y, img_cb, img_cr))
img_ycbcr = np.transpose(img_ycbcr, (1, 2, 0))
img_ycbcr_view = img_ycbcr.astype('uint8')
img_ycbcr_view = Image.fromarray(img_ycbcr_view, mode='YCbCr')
img_ycbcr_view.show()

img_y = upsample(img_ycbcr[...,0], w)
img_cb = upsample(img_ycbcr[...,1], w)
img_cr = upsample(img_ycbcr[...,2], w)

img_ycbcr = np.stack((img_y, img_cb, img_cr))
img_ycbcr = np.transpose(img_ycbcr, (1, 2, 0))
img_ycbcr_view = img_ycbcr.astype('uint8')
img_ycbcr_view = Image.fromarray(img_ycbcr_view, mode='YCbCr')
img_ycbcr_view.show()

print(img_ycbcr.shape)