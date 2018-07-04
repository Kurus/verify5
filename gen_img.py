# this is for float hardware file generation for actual squeezeent
import numpy as np
from scipy import signal as sg
import os
import scipy.io



###########quantization weight###########
from ctypes import *

def d2b(x):
    x = cast(pointer(c_double(x)), POINTER(c_int64)).contents.value
    # print(hex(x))
    x = x+0x0000200000000000
    e = ((x&0x7FF0000000000000)>>52) - 1008
    man = x&0x000C000000000000
    sgn = x&0x8000000000000000
    if e<0:
        e = 0
        man = 0
        sgn = 0
    if e>31:
        e = 31
    bits = sgn>>56 | e<<2 | man>> 50# 64-8,2,52-2
    return np.uint8(bits)
d2bv = np.vectorize(d2b)

def b2d(x):
    ee = ((x&0x7c)>>2)+1008
    if((x&0x7c) == 0):
        if x&0x3==0:
            ee=0
        else:
            ee=1008
    bits = ((x&0x80)<<56) | (ee)<<52 | (x&0x3)<<50
    # print(hex(bits))
    return cast(pointer(c_int64(bits)), POINTER(c_double)).contents.value
b2dv = np.vectorize(b2d) 

#################################################### input image
weights_raw = scipy.io.loadmat("sqz_full.mat")

cwd = os.getcwd()
path = cwd + "/bin"
os.makedirs(path, exist_ok=True)
os.chdir(path)

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    # img_out[:, :, 0] = swap_img[:, :, 2]
    # img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

# import sys
# path = sys.argv[1]
# print(path)
# path='../parrot.jpeg'
# path='../orangutan.jpg'
path='../peacock.jpeg'
img_orig = scipy.misc.imread(path)
img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
mean_pixel = np.array([104.006, 116.669, 122.679])
img=preprocess(img,mean_pixel)
img=d2bv(img)


dim,dim,dep = img.shape

in_ori_c = [] # for first layer in hardware it need to be mod 4
dim_c = dim
if dim%4 ==0:
    dim_c = dim
else:
    dim_c = ((dim//4) + 1)*4
in_ori_c = np.full(dim_c*dim_c*dep, 0, dtype='uint8').reshape((dim_c,dim_c,dep))
in_ori_c[0:dim,0:dim,:] = img

# f_in_c = open("input_layer_c.txt","w")
f_in_c_b = open("input_layer_c.bin","wb")
for d in range(0,dep):
    for z in range(0,dim_c):
        for y in range(0,dim_c):
            lis = in_ori_c[z,y,d].flatten().tolist()
            # f_in_c.write(str(lis)[1:-1]+'\n')
            f_in_c_b.write(bytearray(lis))
f_in_c_b.close()