#from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import cv2 
raw_path = 'ov10642_With_Noise.raw'
#raw_path = 'ov10642_Without_Noise.raw'
from model.dpc import DPC
from model.blc import BLC
from model.aaf import AAF
from model.awb import WBGC
from model.cnf import CNF
from model.cfa import CFA
from model.gac import GC
from model.csc import CSC
from model.nlm import NLM
from model.dwt import DWT
import time
def isp_pipeline(rawimg,raw_w,raw_h):

    dpc_thres = 30
    dpc_mode = 'gradient'
    dpc_clip = 4095
        
    bl_r = 0
    bl_gr = 0
    bl_gb = 0
    bl_b = 0
    alpha = 0
    beta = 0
    blc_clip = 4095
    bayer_pattern = 'rggb'
    
    r_gain = 1.0
    gr_gain = 1.0
    gb_gain = 1.0
    b_gain = 1.0
    awb_clip = 4095
    cfa_mode = 'malvar'
    cfa_clip = 4095
    csc = cp.zeros((3, 4))
    
    csc[0][0] = 1024 * float(0.257) 
    csc[0][1] = 1024 * float(0.504) 
    csc[0][2] = 1024 * float(0.098) 
    csc[0][3] = 1024 * float(16) 
    csc[1][0] = 1024 * float(-0.148) 
    csc[1][1] = 1024 * float(-0.291) 
    csc[1][2] = 1024 * float(0.439) 
    csc[1][3] = 1024 * float(128) 
    csc[2][0] = 1024 * float(0.439) 
    csc[2][1] = 1024 * float(-0.368) 
    csc[2][2] = 1024 * float(-0.071)
    csc[2][3] = 1024 * float(128)
    nlm_h = 10
    nlm_clip = 255
    st = time.time()
    dpc = DPC(rawimg, dpc_thres, dpc_mode, dpc_clip)
    rawimg_dpc = dpc.execute()

    parameter = [bl_r, bl_gr, bl_gb, bl_b, alpha, beta]
    blc = BLC(rawimg_dpc, parameter, bayer_pattern, blc_clip)
    rawimg_blc = blc.execute()
    #print(rawimg_blc.shape)

    # anti-aliasing filter
    aaf = AAF(rawimg_blc)
    rawimg_aaf = aaf.execute()
    
    # white balance gain control
    parameter = [r_gain, gr_gain, gb_gain, b_gain]
    awb = WBGC(rawimg_aaf, parameter, bayer_pattern, awb_clip)
    rawimg_awb = awb.execute()

   
    # chroma noise filtering
    cnf = CNF(rawimg_awb, bayer_pattern, 0, parameter, 1023)
    rawimg_cnf = cnf.execute()
    
    # color filter array interpolation
    cfa = CFA(rawimg_awb, cfa_mode, 'rccc', cfa_clip)
    rgbimg_cfa = cfa.execute()


    # gamma correction
    # look up table
    
    #bw = 12
    #gamma = 0.5
    mode = 'rgb'

    #maxval = pow(2,bw)
    #ind = range(0, maxval)
    #val = [round(pow(float(i)/maxval, gamma) * maxval) for i in ind]
    #lut = dict(zip(ind, val))
    #print(lut)
    #print(ind, val, lut)
    
    gc = GC(rgbimg_cfa, mode)
    rgbimg_gc = gc.execute()
    
 
    #rgbimg_gc = cp.asarray(rgbimg_gc)
    # color space conversion
    
    csc = CSC(rgbimg_gc, csc)
    yuvimg_csc = csc.execute()


    #plt.imshow(yuvimg_csc, cmap='gray')
    #plt.show()
    #bgr = cv2.cvtColor(yuvimg_csc, cv2.COLOR_YUV2BGR);
    #cv2.imwrite('output.png', bgr)
    
    # non-local means denoising
    #nlm = NLM(yuvimg_csc[:,:,0], 1, 4, nlm_h, nlm_clip)
    #yuvimg_nlm = nlm.execute()
    
    #print(yuvimg_csc[:,:,1])
    
    dwt = DWT(yuvimg_csc[:,:,0])
    #show = yuvimg_csc[:,:,0].astype(cp.uint8)
    yuvimg_dwt = dwt.execute()
    y_img_dwt = (yuvimg_dwt)
    
    et = time.time()
    res = et - st
    final_res = res * 1000 
    print('Execution time:', final_res, 'milliseconds')   
    print('FPS:', 1000/final_res)
    
    cv2.imshow('cv', y_img_dwt.get())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_dwt.jpg', y_img_dwt.get())
    
    #et = time.time()
    #res = et - st
    #final_res = res * 1000 
    #print('Execution time:', final_res, 'milliseconds')
    #plt.imshow(yuvimg_nlm, cmap='gray')
    #plt.show()    
    yuvimg_out = np.empty((raw_h, raw_w, 3), dtype=np.uint8)

    #yuvimg_out[:,:,0] = yuvimg_nlm.get()
    yuvimg_out[:,:,0] = yuvimg_csc[:,:,0].get()
    yuvimg_out[:,:,1:3] = yuvimg_csc[:,:,1:3].get()

    img_bgr = cv2.cvtColor(yuvimg_out, cv2.COLOR_YCrCb2BGR)
    return img_bgr
    
if __name__ == '__main__':
    raw_w = 1280
    raw_h = 720

    rawimg = cp.fromfile(raw_path, dtype='uint16', sep='')
    rawimg = rawimg.reshape([raw_h, raw_w])
    #print(rawimg)
    #print(rawimg.shape)
    done = isp_pipeline(rawimg,raw_w,raw_h)
    
    cv2.imshow('cv', (done/256))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('output.jpg', done)