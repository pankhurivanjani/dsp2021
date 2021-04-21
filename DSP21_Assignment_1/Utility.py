## Split input matrix "img" to submatrices of size "b" and
## apply the given function fun to each submatrices.
## Return a matrix "out" out of all submatrices. 
## Usage example: output = blockproc(input, [10, 10], dct2)
def blockproc(img, b, fun):
    import numpy as np
    x,y = img.shape
    bx = b[0]
    by = b[1]
    
    out = np.zeros((x,y))
    #print(out)
    for i in range(0,x,bx):
        for j in range(0,y,by):
            #print (img[i:i+bx, j:j+by])
            out[i:i+bx, j:j+by] = fun(img[i:i+bx,j:j+by])
    
    return out
