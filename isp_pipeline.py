#from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import cv2 
import glob, os
raw_path = 'ov10642_With_Noise_2023.01.018.raw'
#raw_path = 'ov10642_Without_Noise_2023.01.017.raw'

#raw_path = 'ov10642_Without_Noise_2023.01.04.raw'
#raw_path = 'ov10642_With_Noise_2023.01.04.raw'
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
import rawpy
import time
import sys
import imageio

def isp_pipeline(rawimg,raw_w,raw_h,gamma=0.5,bl=[0,0,0,0],dpc_thres=4095,r_gain=1.0,gr_gain=1.0,gb_gain=1.0,b_gain=1.0,nr='enable',bayer_pattern = 'rggb',bandwidth_bit=12,save_picture='disable'):

    dpc_thres = dpc_thres
    dpc_mode = 'gradient'
    maxval = pow(2,bandwidth_bit)-1
    dpc_clip = maxval
    print(maxval)
    bl_r = bl[0]
    bl_gr = bl[1]
    bl_gb = bl[2]
    bl_b = bl[3]
    alpha = 0.0
    beta = 0.0
    blc_clip = maxval
    #bayer_pattern = 'rggb'
    
    r_gain = r_gain
    gr_gain = gr_gain
    gb_gain = gb_gain
    b_gain = b_gain
    awb_clip = maxval
    cfa_mode = 'malvar'
    cfa_clip = maxval
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
    nlm_clip = maxval
    tmp_img = rawimg/maxval
    cv2.imwrite('source.jpg', (tmp_img*255).get()) 
    st = time.time()
    dpc = DPC(rawimg, dpc_thres, dpc_mode, dpc_clip, bayer_pattern)
    rawimg_dpc = dpc.execute()
    if save_picture=='enable':
        tmp_img = rawimg_dpc/maxval
        cv2.imwrite('rawimg_dpc.jpg', (tmp_img*255).get()) 
    
    parameter = [bl_r, bl_gr, bl_gb, bl_b, alpha, beta]
    blc = BLC(rawimg_dpc, parameter, bayer_pattern, blc_clip)
    rawimg_blc = blc.execute()
    if save_picture=='enable':
        tmp_img = rawimg_blc/maxval
        cv2.imwrite('rawimg_blc.jpg', (tmp_img*255).get()) 
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
    if save_picture=='enable':
        tmp_img = rawimg_awb/maxval
        cv2.imwrite('rawimg_awb.jpg', (tmp_img*255).get())    
    #tmp_img = rawimg_awb/4096
    
    #cv2.imshow('cv', tmp_img.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # chroma noise filtering
    #cnf = CNF(rawimg_awb, bayer_pattern, 0, parameter, maxval)
    #rawimg_cnf = cnf.execute()
    
    # color filter array interpolation
    #cfa = CFA(rawimg_awb, cfa_mode, 'bggr', cfa_clip)
    cfa = CFA(rawimg_awb, cfa_mode, bayer_pattern, cfa_clip,maxval)
    rgbimg_cfa = cfa.execute()
    if save_picture=='enable':
        tmp_img = rgbimg_cfa/maxval
        cv2.imwrite('rawimg_cfa.jpg', (tmp_img*255).get())        
    
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
    divide = pow(2,int((bandwidth_bit-8)))
    if gamma!=1.0:
        gc = GC(rgbimg_cfa, mode,gamma=gamma,bandwidth_bit=bandwidth_bit)
        rgbimg_gc = gc.execute()
    else:      
        rgbimg_gc = rgbimg_cfa/divide
    if save_picture=='enable':
        tmp_img = rgbimg_gc/256
        cv2.imwrite('rawimg_gc.jpg', (tmp_img*255).get())    
    #rgbimg_gc[...,0] = rgbimg_gc[...,0]
    #rgbimg_gc[...,1] = rgbimg_gc[...,1]
    #rgbimg_gc[...,2] = rgbimg_gc[...,2]
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
    if nr == 'enable':
        print('test')
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
    if nr == 'enable':
        yuvimg_out[:,:,0] = yuvimg_dwt[:,:,0].get()       
        yuvimg_out[:,:,1:3] = yuvimg_dwt[:,:,1:3].get()
    else:
        yuvimg_out[:,:,0] = yuvimg_csc[:,:,0].get()
        yuvimg_out[:,:,1:3] = yuvimg_csc[:,:,1:3].get()
        
    img_bgr = cv2.cvtColor(yuvimg_out, cv2.COLOR_YCrCb2BGR)
    
    return img_bgr
def process_single_image(file,dir):
    rawImg = rawpy.imread(os.path.join(dir,file))
    filename,ext = os.path.splitext(file)
    rawimg = rawImg.raw_image_visible
    #print(os.path.join(dir,filename+'.jpg'))
    
    r_gain = rawImg.camera_whitebalance[0]/1024
    gr_gain = rawImg.camera_whitebalance[1]/1024
    b_gain = rawImg.camera_whitebalance[2]/1024
    gb_gain = rawImg.camera_whitebalance[1]/1024
    
    #print(rawImg.camera_white_level_per_channel)
    
    #rawimg = RGB_to_Bayer(path=sys.argv[1])
    rawimg = cp.asarray(rawimg)
    rawimg = rawimg[:3464,:5200]
    raw_h = rawimg.shape[0]
    raw_w = rawimg.shape[1]
    #print(raw_h,raw_w,rawImg.daylight_whitebalance)
    done = isp_pipeline(rawimg,raw_w,raw_h,r_gain=r_gain*4,b_gain=b_gain*4,gr_gain=gr_gain*4,gb_gain=gb_gain*4,bl=-2047,gamma=0.7,nr='enable',bayer_pattern='gbrg')
    cv2.imwrite(os.path.join(dir,filename+'.jpg'), done)    
        
def batch(dir="T3i_RAW") :
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.endswith(".CR2"):
                print(os.path.join(root, file))
                process_single_image(file,root)
                #return

if __name__ == '__main__':
    # Try to delete the file.
    if os.path.isfile('output.jpg'):
        os.remove('output.jpg')
    if len(sys.argv) ==1:
        rawimg = RGB_to_Bayer()
        raw_h = rawimg.shape[0]
        raw_w = rawimg.shape[1]
        done = isp_pipeline(rawimg,raw_w,raw_h,gamma=1.0,nr='disable',bayer_pattern='ccrc')        
    elif sys.argv[1]=='raw' :
        raw_w = 1280
        raw_h = 1080
        rawimg = cp.fromfile(raw_path, dtype='uint16', sep='')
        rawimg = rawimg.reshape([raw_h, raw_w])
        print(cp.amin(rawimg))
        ae_gain = 1.0
        r_gain = 2.0*ae_gain
        gr_gain = 1.0*ae_gain
        gb_gain = 1.0*ae_gain
        b_gain = 1.0*ae_gain
        done = isp_pipeline(rawimg,raw_w,raw_h,gamma=0.5,bl=[-0,-0,-0,-0],dpc_thres=100,
            r_gain=r_gain,gr_gain=gr_gain,gb_gain=gb_gain,b_gain=b_gain,nr='enable',bayer_pattern='ccrc',save_picture='disable')  
    elif 'batch' in sys.argv[1]:
        rawImg = rawpy.imread(sys.argv[1])
        batch()
    else :
        #rawimg = rawImg.raw_image_visible
        #rgb = rawImg.postprocess()
        #imageio.imsave('default.tiff', rgb)
        
        r_gain = rawImg.camera_whitebalance[0]/1024
        gr_gain = rawImg.camera_whitebalance[1]/1024
        b_gain = rawImg.camera_whitebalance[2]/1024
        gb_gain = rawImg.camera_whitebalance[1]/1024
        
        print(rawImg.camera_whitebalance)
        
        #rawimg = RGB_to_Bayer(path=sys.argv[1])
        rawimg = cp.asarray(rawimg)
        #rawimg = rawimg[:3200,:5200]
        raw_h = rawimg.shape[0]
        raw_w = rawimg.shape[1]
        print(raw_h,raw_w,rawImg.daylight_whitebalance)
        #done = isp_pipeline(rawimg,raw_w,raw_h,r_gain=r_gain,b_gain=b_gain,gr_gain=gr_gain,gb_gain=gb_gain,bl=-2000,gamma=0.45,nr='disable',bayer_pattern='gbrg')  
        
    
    #cv2.imshow('cv', (done/256))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()   
    cv2.imwrite('output.jpg', done)       