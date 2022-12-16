#from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import cv2 
raw_path = 'ov10642_With_Noise.raw'
#raw_path = 'ov10642_Without_Noise.raw'
#raw_path = 'DSCF0173_H_MTF.bmp_(1280x960_12)_LIT_ENDIAN_RGGB.raw'
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
from tool import RGB_to_Bayer
import time
import sys
def isp_pipeline(rawimg,raw_w,raw_h,gamma=0.5,bl=0,dpc_thres=4095,r_gain=1.0,gr_gain=1.0,gb_gain=1.0,b_gain=1.0):

    dpc_thres = dpc_thres
    dpc_mode = 'gradient'
    dpc_clip = 4095
        
    bl_r = bl
    bl_gr = bl
    bl_gb = bl
    bl_b = bl
    alpha = 0.0
    beta = 0.0
    blc_clip = 4095
    bayer_pattern = 'rggb'
    
    r_gain = r_gain
    gr_gain = gr_gain
    gb_gain = gb_gain
    b_gain = b_gain
    awb_clip = 4095
    cfa_mode = 'malvar'
    cfa_clip = 4095
    csc = cp.zeros((3, 4))
    
    csc[0][0] = 1024 * float(0.299) 
    csc[0][1] = 1024 * float(0.587) 
    csc[0][2] = 1024 * float(0.114) 
    csc[0][3] = 1024 * float(0) 
    csc[1][0] = 1024 * float(-0.169) 
    csc[1][1] = 1024 * float(-0.331) 
    csc[1][2] = 1024 * float(0.5) 
    csc[1][3] = 1024 * float(128) 
    csc[2][0] = 1024 * float(0.5) 
    csc[2][1] = 1024 * float(-0.419) 
    csc[2][2] = 1024 * float(-0.081)
    csc[2][3] = 1024 * float(128)
    nlm_h = 10
    nlm_clip = 255
    #tmp_img = rawimg/4095
    #cv2.imwrite('source.jpg', (tmp_img*255).get()) 
    st = time.time()
    dpc = DPC(rawimg, dpc_thres, dpc_mode, dpc_clip)
    rawimg_dpc = dpc.execute()
    #tmp_img = rawimg_dpc/4095
    #cv2.imwrite('rawimg_dpc.jpg', (tmp_img*255).get()) 
    parameter = [bl_r, bl_gr, bl_gb, bl_b, alpha, beta]
    blc = BLC(rawimg_dpc, parameter, bayer_pattern, blc_clip)
    rawimg_blc = blc.execute()
    #tmp_img = rawimg_blc/4095
    #cv2.imwrite('rawimg_blc.jpg', (tmp_img*255).get()) 
    #print(rawimg_blc.shape)
    #tmp_img = rawimg_blc/4096

    #cv2.imshow('cv', tmp_img.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('output_tmp1.jpg', (tmp_img*255).get())
    # anti-aliasing filter
    #aaf = AAF(rawimg_blc)
    #rawimg_aaf = aaf.execute()
    
    #tmp_img = rawimg_aaf/1024

    #cv2.imshow('cv', tmp_img.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('output_tmp2.jpg', (tmp_img*255).get())    
    # white balance gain control
    parameter = [r_gain, gr_gain, gb_gain, b_gain]
    awb = WBGC(rawimg_blc, parameter, bayer_pattern, awb_clip)
    rawimg_awb = awb.execute()
    #tmp_img = rawimg_awb/4095
    #cv2.imwrite('rawimg_awb.jpg', (tmp_img*255).get())    
    #tmp_img = rawimg_awb/4096

    #cv2.imshow('cv', tmp_img.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
   
    # chroma noise filtering
    #cnf = CNF(rawimg_awb, bayer_pattern, 0, parameter, 1023)
    #rawimg_cnf = cnf.execute()
    
    # color filter array interpolation
    #cfa = CFA(rawimg_awb, cfa_mode, 'bggr', cfa_clip)
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
    if gamma!=1.0:
        gc = GC(rgbimg_cfa, mode,gamma=gamma)
        rgbimg_gc = gc.execute()
    else:
        rgbimg_gc = rgbimg_cfa/16
 
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
    
    dwt = DWT(yuvimg_csc)
    #show = yuvimg_csc[:,:,0].astype(cp.uint8)
    yuvimg_dwt = dwt.execute()
    #print(yuvimg_dwt.shape)
    
    et = time.time()
    res = et - st
    final_res = res * 1000 
    
    print('Execution time:', final_res, 'milliseconds')   
    print('FPS:', 1000/final_res)
    

    
    #tmp_img = rgbimg_cfa/4096

    #cv2.imshow('cv', tmp_img.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('output_tmp.jpg', (tmp_img*255).get())   
    #et = time.time()
    #res = et - st
    #final_res = res * 1000 
    #print('Execution time:', final_res, 'milliseconds')
    #plt.imshow(yuvimg_nlm, cmap='gray')
    #plt.show()    
    yuvimg_out = np.empty((raw_h, raw_w, 3), dtype=np.uint8)

    #yuvimg_out[:,:,0] = yuvimg_nlm.get()
    yuvimg_out[:,:,0] = yuvimg_dwt[:,:,0].get()
    #yuvimg_out[:,:,0] = yuvimg_csc[:,:,0].get()
    yuvimg_out[:,:,1:3] = yuvimg_dwt[:,:,1:3].get()

    img_bgr = cv2.cvtColor(yuvimg_out, cv2.COLOR_YCrCb2BGR)
    return img_bgr
    
if __name__ == '__main__':
    
    if len(sys.argv) ==1:
        rawimg = RGB_to_Bayer()
        raw_h = rawimg.shape[0]
        raw_w = rawimg.shape[1]
        done = isp_pipeline(rawimg,raw_w,raw_h,gamma=1.0)        
    elif sys.argv[1]=='raw' :
        raw_w = 1280
        raw_h = 720
        rawimg = cp.fromfile(raw_path, dtype='uint16', sep='')
        rawimg = rawimg.reshape([raw_h, raw_w])
        done = isp_pipeline(rawimg,raw_w,raw_h,gamma=0.5,bl=-60,dpc_thres=30,r_gain=2.0,gr_gain=2.0,gb_gain=4.0,b_gain=2.0)  
    else:
        rawimg = RGB_to_Bayer(path=sys.argv[1])
        raw_h = rawimg.shape[0]
        raw_w = rawimg.shape[1]
        done = isp_pipeline(rawimg,raw_w,raw_h,gamma=1.0)  
        
    
    #cv2.imshow('cv', (done/256))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()   
    cv2.imwrite('output.jpg', done)       