# 2.1
import numpy as np
import pdb
from PIL import Image
from matplotlib import pyplot as plt

vec1 = np.array([0, 127.5, 127.5])
mat1 = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

def imshow(img, mode='YCbCr'): # img - PIL image object
    '''
    img = img.astype('uint8')
    img = Image.fromarray(img, mode=mode)
    img.show()
    '''
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

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
    #pdb.set_trace()
    out1 = dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')
    '''x, y = arr.shape
    out = np.zeros((x,y))
    for i in range(0, x):
        out[i] = dct(arr[i], norm='ortho')
    for j in range(0, y):
        out[:,j] = dct(out[:, j], norm='ortho')
    #print(out1-out)
    '''
    return out1

def idct2(arr):
    out2 = idct(idct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')
    '''x, y = arr.shape
    out = np.zeros((x,y))
    for i in range(0, x):
        out[i] = idct(arr[i], norm='ortho')
    for j in range(0, y):
        out[:,j] = idct(out[:, j], norm='ortho')'''
    return out2

# 2.4 

Q = np.array([[8, 19, 26, 29], 
            [19, 26, 29, 34], 
            [22, 27, 32, 40],
            [26, 29, 38, 56]])

'''
Q = np.array([[10, 15, 25, 37, 51, 66, 82, 100],
            [15, 19, 28, 39, 52, 67, 83, 101],
            [25, 28, 35, 45, 58, 72, 88, 105],
            [37, 39, 45, 54, 66, 79, 94, 111],
            [51, 52, 58, 66, 76, 89, 103, 119],
            [66, 67, 72, 79, 89, 101, 114, 130],
            [82, 83, 88, 94, 103, 114, 127, 142],
            [100, 101, 105, 111, 119, 130, 142, 156]])
'''
'''
Q = np.array([[80,60,50,80,120,200,255,255],
            [55,60,70,95,130,255,255,255],
            [70,65,80,120,200,255,255,255],
            [70,85,110,145,255,255,255,255],
            [90,110,185,255,255,255,255,255],
            [120,175,255,255,255,255,255,255],
            [245,255,255,255,255,255,255,255],
            [255,255,255,255,255,255,255,255]])

'''
def quanmat(M):
    #pdb.set_trace()
    x, y = M.shape
    out = np.zeros((x,y))
    f = M.shape[0] // Q.shape[0]
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            out[i*f: (i+1)*f, j*f: (j+1)*f] =  M[i*f:(i+1)*f, j*f:(j+1)*f] / Q[i,j] 
    #pdb.set_trace()
    return out
'''
def quanmat(M):
    #pdb.set_trace()
    #x, y = M.shape
    #out = np.zeros((x,y))
    #f = M.shape[0] // Q.shape[0]
    #pdb.set_trace()
    myout = np.around(M / Q)
    return myout
'''
def dequanmat(M):
    x, y = M.shape
    out = np.zeros((x,y))
    f = M.shape[0] // Q.shape[0]
    #f = 8
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            out[i*f: (i+1)*f, j*f: (j+1)*f] =  M[i*f:(i+1)*f, j*f:(j+1)*f] * Q[i,j] 
    return out
'''

def dequanmat(M):
    out = M * Q
    return out
'''

# 2.5
from Utility import blockproc

img_y = img_ycbcr[...,0]
print(img_y)
#imshow(img_y, mode='P') 
img_y = np.float32(img_y)
#img_y =  img_y - 128.
#img_y_dct = blockproc(img_y, [8, 8], dct2)
img_y_dct = np.zeros(img_y.shape)

# Do 8x8 DCT on image (in-place)
for i in np.r_[:img_y.shape[0]:8]:
    for j in np.r_[:img_y.shape[1]:8]:
        img_y_dct[i:(i+8),j:(j+8)] = dct2( img_y[i:(i+8),j:(j+8)] )

#pdb.set_trace()
imshow(img_y_dct, mode='P') 
img_y_dct_idct = blockproc(img_y_dct, [8, 8], idct2)
print(np.linalg.norm(img_y - img_y_dct_idct))

#pdb.set_trace()
img_y_dct_q = blockproc(img_y_dct, [8, 8], quanmat)
#imshow(img_y_dct_q, mode='P') 

img_y_dct_q_dq = img_y_dct_q#blockproc(img_y_dct_q, [8, 8], dequanmat) #img_y_dct_q#
#print(np.linalg.norm(img_y_dct - img_y_dct_q_dq))

img_y_dct_q_dq_idct = blockproc(img_y_dct_q_dq, [8, 8], idct2)
#pdb.set_trace()
#imshow(img_y_dct_q_dq_idct, mode='P') 
print(np.linalg.norm(img_y - img_y_dct_q_dq_idct))
#img_y = img_y + 128
#img_y_dct_q_dq_idct =  img_y_dct_q_dq_idct + 128
#pdb.set_trace()
#img_y_dct_q_dq_idct *= 255.0/image.max()img_y_dct_q_dq_idct
img_y_dct_q_dq_idct = 255 * (img_y_dct_q_dq_idct - img_y_dct_q_dq_idct.min()) /img_y_dct_q_dq_idct.max()
print(np.linalg.norm(img_y - img_y_dct_q_dq_idct))
pdb.set_trace()
imshow(img_y_dct_q_dq_idct, mode='P') 

stacked = np.hstack((img_y, img_y_dct_q_dq_idct, img_y - img_y_dct_q_dq_idct))
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


