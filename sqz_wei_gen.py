# this is for float hardware file generation for actual squeezeent
import numpy as np
from scipy import signal as sg
import os
import scipy.io

sq_rep_list = [0,0,0,0,0,0,0,0,0] # repete squze kernl for last layer

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
# if cur_ly == 0:
#     #######################         Input image
#     if random == 0:
#         in_ori = np.full(dim*dim*dep, 0, dtype='uint8').reshape((dim,dim,dep))
#         in_ori[:,:,0] = np.arange(dim*dim, dtype = 'uint8').reshape(dim,dim)
#         in_ori = np.arange(dim*dim*dep, dtype='uint8').reshape((dim,dim,dep))
#     else:
#         in_ori = np.random.randint(low = 0, high = 255, size = (dim,dim,dep), dtype='uint8')
# else:
#     in_ori = d2bv(np.rollaxis(final_out,0,3))
#     dim,_,dep = in_ori.shape
#     dim_p=dim + 2

# in_l = np.zeros(dim_p*dim_p*dep, dtype='uint8').reshape((dim_p,dim_p,dep))
# in_l[1:-1,1:-1,:] = in_ori
# print("input layer");print(in_l[:,:,0]);
# # f_in = open("input_layer.txt","w")
# # f_in_b = open("input_layer.bin","wb")
# # for z in range(0,dim):
# #     for y in range(0,dep):
# #         for x in range(0,dim):
# #             lis = in_l[z:z+3,x:x+3,y].flatten().tolist()
# #             for rep in range(0,ker,4):
# #                 f_in.write(str(lis)[1:-1]+'\n')
# #                 f_in_b.write(bytearray(lis))

################################################## squeezent
for i in range(1,10):
    cur_ly = i-1
    sq_rep = sq_rep_list[cur_ly]
    ############# geting weight from mat file
    e1="fire"+str(i)+"/expand1x1"
    e3="fire"+str(i)+"/expand3x3"
    sq="fire"+str(i+1)+"/squeeze1x1"
    if i==1:
        e1=""
        e3="conv1"

        #for conv1 swap
        exp3k, exp3b = weights_raw[e3][0]
        tmp0 = np.array(exp3k[:,:,0,:])
        tmp1 = np.array(exp3k[:,:,2,:])
        exp3k[:,:,2,:] = tmp0
        exp3k[:,:,0,:] = tmp1
        sqk, sqb = weights_raw[sq][0]
        sqk = np.concatenate((np.zeros(sqk.shape), sqk), axis=2)
        # conv1 exp 1 is zeros
        ektmp = (1,1)+exp3k.shape[2:]
        exp1k=np.zeros(ektmp)
        exp1b=np.zeros(exp3b.shape)
    else:
        if i==9:
            sq="conv10"
        exp1k, exp1b = weights_raw[e1][0]
        exp3k, exp3b = weights_raw[e3][0]
        sqk, sqb = weights_raw[sq][0]
    exp1k=np.moveaxis(exp1k,[-1,-2,0,1],[0,1,2,3])
    exp1k=exp1k.squeeze()
    exp1b=exp1b.squeeze()
    exp3k=np.moveaxis(exp3k,[-1,-2,0,1],[0,1,2,3])
    ektmp = exp3k.shape[:-2]+(9,)
    exp3k = exp3k.reshape(ektmp)
    exp3b=exp3b.squeeze()
    sqk = sqk.squeeze()
    sqb = sqb.squeeze()

    exp1k=d2bv(exp1k)
    exp1b=d2bv(exp1b)
    exp3k=d2bv(exp3k)
    exp3b=d2bv(exp3b)
    sqk  =d2bv(sqk)
    sqb  =d2bv(sqb)
    ################ exp kernels
    ker,dep = exp1k.shape
    ker_l_1 = exp1k
    # ker_l_1 = np.zeros(ker*dep, dtype='uint8').reshape((ker,dep))
    print("kernel1");print(ker_l_1)
    # f_k_1 = open("ker_1x1.txt","w")
    f_k_1_b = open("ker_1x1"+"_"+str(cur_ly)+".bin","wb")
    for z in range(0,dep):
        for x in range(0,ker,8):
            lis = ker_l_1[x:x+4,z][::-1]
            f_k_1_b.write(bytearray(lis))
            # f_k_1.write(str(lis)[1:-1]+'\n')

            lis = ker_l_1[x+4:x+8,z][::-1]
            f_k_1_b.write(bytearray(lis))
        # f_k_1.write(str(lis)[1:-1]+'\n')
    f_k_1_b.close()
    ker_l_3 = exp3k
    # ker_l_3 = np.random.randint(low = 0, high = 255, size = (ker,dep,9),dtype='uint8').reshape((ker,dep,9))
    # f_k_3 = open("ker_3x3.txt","w")
    f_k_3_b = open("ker_3x3"+"_"+str(cur_ly)+".bin","wb")
    # for m in range(0,dim): # repet 3x3 kernel # removed repeating
    # ordering 78 345 012 ## 6
    for z in range(0,dep):
        lis = ker_l_3[:,z,:]
        for x in range(0,ker,8):
            for a in range(0,8):
                eig = lis[x+a,[7,8,3,4,5,0,1,2]] #reversed
                f_k_3_b.write(bytearray(eig))
                # f_k_3.write(str(eig)[1:-1]+'\n')
            nin = lis[x:x+8,6].flatten() #no reversed 6 means 
            f_k_3_b.write(bytearray(nin))
            # f_k_3.write(str(nin)[1:-1]+'\n')
    f_k_3_b.close()
    ########################        exapnd bias
    bis_1 = exp1b
    # bis_1 = np.full(ker,0x00,dtype='uint8') #one
    bis_3 = exp3b
    # bis_3 = np.full(ker,0x00,dtype='uint8')
    # b_bis = open("bias.txt","w")
    b_bis_b = open("bias"+"_"+str(cur_ly)+".bin","wb")
    for i in range(0,ker,4):
        lis_b3 = bis_3[i:i+4][::-1] # reverse
        lis_b1 = bis_1[i:i+4][::-1] #reverst
        # b_bis.write(str(lis_b1)[1:-1]+'\n')
        # b_bis.write(str(lis_b3)[1:-1]+'\n')
        b_bis_b.write(bytearray(lis_b1))
        b_bis_b.write(bytearray(lis_b3))
    b_bis_b.close()
    ########################   squ kernel
    sqk = np.moveaxis(sqk,0,-1)
    sq_ker,dep = sqk.shape
    # sq_ker_l = np.full(sq_ker*dep,0, dtype='uint8').reshape((sq_ker,dep))
    sq_ker_l = sqk
    # sq_k_1 = open("sq_ker.txt","w")
    sq_k_1_b = open("sq_ker"+"_"+str(cur_ly)+".bin","wb")
    dep_h = dep//2

    rep_no = 1
    if(sq_rep == 1):
        rep_no = dim_sq
    for r in range(0,rep_no):
        for x in range(0,sq_ker):
            for z in range(0,dep_h,8):
                lis = sq_ker_l[x,z+dep_h:z+dep_h+8][::-1]#kerle of 3x3 part # reverse added
                # sq_k_1.write(str(lis)[1:-1]+'\n')
                sq_k_1_b.write(bytearray(lis))

                lis = sq_ker_l[x,z:z+8][::-1] #reverse added
                # sq_k_1.write(str(lis)[1:-1]+'\n')
                sq_k_1_b.write(bytearray(lis))
    sq_k_1_b.close()
    sq_ker_l = b2dv(sq_ker_l) #########converting to float
    print("sqeeze kernel");print(sq_ker_l[0,:])
    # #######################    squ bias
    sq_bis_1 = sqb
    # sq_bis_1 = np.random.randint(low = 0, high = 255, size = (sq_ker),dtype='uint8')
    # print(sq_bis_1)
    # f_sq_bis = open("sq_bias.txt","w")
    f_sq_bis_b = open("sq_bias"+"_"+str(cur_ly)+".bin","wb")
    for x in range(0,sq_ker,8):
        lis = sq_bis_1[x:x+8]
        # lis = lis[::-1] #reverse
        # f_sq_bis.write(str(lis)[1:-1]+'\n')
        f_sq_bis_b.write(bytearray(lis))
    f_sq_bis_b.close()
os.chdir(cwd)