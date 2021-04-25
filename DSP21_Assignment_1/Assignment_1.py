# 2.1
import numpy as np
import pdb
from PIL import Image
from matplotlib import pyplot as plt

vec1 = np.array([0, 127.5, 127.5])
mat1 = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

def imshow(img, mode='YCbCr'): # img - PIL image object
    img = img.astype('uint8')
    img = Image.fromarray(img, mode=mode)
    img.show()
    '''
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()
    '''

def rgb2ycbcr(rgb):
    ycbcr = vec1 + rgb.dot(mat1.T) 
    return ycbcr

def ycbcr2rgb(ycbcr):
    mat1_inv = np.linalg.inv(mat1)
    rgb = (ycbcr - vec1).dot(mat1_inv.T)
    return rgb

img = Image.open("birds.ppm")
img_rgb = np.asarray(img) #default image mode is rgb
#imshow(img_rgb, mode='RGB')

img_ycbcr = rgb2ycbcr(img_rgb)
#imshow(img_ycbcr)

# 2.2
def downsample(img, w):
    p, q = img.shape[0]//w, img.shape[1]//w #math.ceil(x/w), math.ceil(y/w)
    down_out = np.zeros((p, q))
    for i in range(0, p):
        for j in range(0, q):
            down_out[i, j] = np.mean(img[i*w: (i+1)*w, j*w: (j+1)*w])
    return down_out

def upsample(img, w): 
    up_out = img.repeat(w, axis=0).repeat(w, axis=1)
    return up_out

w = 8

img_ycbcr_d = np.dstack((downsample(img_ycbcr[...,0], w), downsample(img_ycbcr[...,1], w), downsample(img_ycbcr[...,2], w)))
print(img_ycbcr_d.shape)
#imshow(img_ycbcr_d)

img_ycbcr_u = np.dstack((upsample(img_ycbcr_d[...,0], w), upsample(img_ycbcr_d[...,1], w), upsample(img_ycbcr_d[...,2], w)))
print(img_ycbcr_u.shape)
#imshow(img_ycbcr_u)

#DCT
from scipy.fft import dct, idct

def dct2(arr):
    return dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(arr):
    return idct(idct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')

# 2.4 

Q = np.array([[8, 19, 26, 29], 
            [19, 26, 29, 34], 
            [22, 27, 32, 40],
            [26, 29, 38, 56]])

def quanmat(M):
    M_quant = M
    f = M.shape[0] // Q.shape[0]
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            M_quant[i*f: (i+1)*f, j*f: (j+1)*f] /= Q[i,j] 
    return np.around(M_quant)

def dequanmat(M):
    M_dequant = M
    f = M.shape[0] // Q.shape[0]
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            M_dequant[i*f: (i+1)*f, j*f: (j+1)*f] *= Q[i,j] 
    return M_dequant

# 2.5
from Utility import blockproc

img_y = img_ycbcr[...,0]
imshow(img_y, mode='P') 

img_y_dct = blockproc(img_y, [w, w], dct2)
imshow(img_y_dct, mode='P') 
#img_y_dct_idct = blockproc(img_y_dct, [w, w], idct2)
#print(np.linalg.norm(img_y - img_y_dct_idct))

img_y_dct_q = blockproc(img_y_dct, [w, w], quanmat)
imshow(img_y_dct_q, mode='P') 
#img_y_dct_q_dq = blockproc(img_y_dct_q, [8, 8], dequanmat) 
#print(np.linalg.norm(img_y_dct - img_y_dct_q_dq))

img_y_dct_q_idct = blockproc(img_y_dct_q, [w, w], idct2)
img_y_dct_q_idct = 255 * (img_y_dct_q_idct - img_y_dct_q_idct.min()) /img_y_dct_q_idct.max()
#imshow(img_y_dct_q_idct, mode='P') 
#print(np.linalg.norm(img_y - img_y_dct_q_idct))

stacked = np.hstack((img_y, img_y_dct_q_idct, img_y - img_y_dct_q_idct))
imshow(stacked, mode='P') 

'''
img_cb_d = downsample(img_ycbcr[...,1], w)
img_cb_d_dct = blockproc(img_cb_d, [8, 8], dct2)

img_cr_d = downsample(img_ycbcr[...,2], w)
img_cr_d_dct = blockproc(img_cr_d, [8, 8], dct2)
'''

def encodemat():
    from dahuffman import HuffmanCodec
    codec = HuffmanCodec.from_data('hello world how are you doing today foo bar lorem ipsum')
    codec.encode('do lo er ad od')
    return codec

def decodemat(codec, encoded):
    decoded = codec.decode(encoded)
    return decoded


img_y_dct_q_dq = img_y_dct_q#blockproc(img_y_dct_q, [8, 8], dequanmat) 