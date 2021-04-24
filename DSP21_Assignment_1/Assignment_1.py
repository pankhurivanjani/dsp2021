# 2.1
import numpy as np
import pdb
from PIL import Image

vec1 = np.array([0, 127.5, 127.5])
mat1 = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

def imshow(img, mode='YCbCr'): # img - PIL image object
    img = img.astype('uint8')
    img = Image.fromarray(img, mode=mode)
    img.show()

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
imshow(img_ycbcr)

# 2.2
def downsample(img, w):
    p, q = img.shape[0]//w, img.shape[1]//w #math.ceil(x/w), math.ceil(y/w)
    out = np.zeros((p, q))
    for i in range(0, p):
        for j in range(0, q):
            out[i, j] = np.mean(img[i*w: (i+1)*w, j*w: (j+1)*w])
    return out

def upsample(img, w): 
    out = img.repeat(w, axis=0).repeat(w, axis=1)
    return out

w = 8

img_ycbcr_d = np.dstack((downsample(img_ycbcr[...,0], w), downsample(img_ycbcr[...,1], w), downsample(img_ycbcr[...,2], w)))
print(img_ycbcr_d.shape)
imshow(img_ycbcr_d)

img_ycbcr_u = np.dstack((upsample(img_ycbcr_d[...,0], w), upsample(img_ycbcr_d[...,1], w), upsample(img_ycbcr_d[...,2], w)))
print(img_ycbcr_u.shape)
imshow(img_ycbcr_u)

#DCT
from scipy.fft import dct, idct

def dct2(arr):
    x, y = arr.shape
    for i in range(0, x):
        arr[i] = dct(arr[i], norm='ortho')
    for j in range(0, y):
        arr[:,j] = dct(arr[:, j], norm='ortho')
    return arr

def idct2(arr):
    x, y = arr.shape
    for i in range(0, x):
        arr[i] = idct(arr[i], norm='ortho')
    for j in range(0, y):
        arr[:,j] = idct(arr[:, j], norm='ortho')
    return arr

# 2.4 
def quanmat(M, Q):
    f = M.shape[0] // Q.shape[0]
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            M[i*f: (i+1)*f, j*f: (j+1)*f] =  M[i*f:(i+1)*f, j*f:(j+1)*f] / Q[i,j] 
    return M

def dequanmat(M, Q):
    f = M.shape[0] // Q.shape[0]
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            M[i*f: (i+1)*f, j*f: (j+1)*f] =  M[i*f:(i+1)*f, j*f:(j+1)*f] * Q[i,j] 
    return M

# 2.5
from Utility import blockproc

Q = [[8, 19, 26, 29], 
    [19, 26, 29, 34], 
    [22, 27, 32, 40],
    [26, 29, 38, 56]]

img_y = img_ycbcr[...,0]
img_y_dct = blockproc(img_y, [8, 8], dct2)
#img_y_idct = blockproc(img_y, [8, 8], idct2)
#print(np.linalg.norm(img_y - img_y_idct))

img_cb_d = downsample(img_ycbcr[...,1], w)
img_cb_d_dct = blockproc(img_cb_d, [8, 8], dct2)

img_cr_d = downsample(img_ycbcr[...,2], w)
img_cr_d_dct = blockproc(img_cr_d, [8, 8], dct2)
