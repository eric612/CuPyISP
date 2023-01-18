#!/usr/bin/python
import numpy as np
import cupy as cp
class GC:
    'Gamma Correction'

    def __init__(self, img, mode,clip=1.0,gamma=0.5,bandwidth_bit=12):
        self.img = img
        #self.lut = lut
        self.mode = mode
        self.clip = clip
        self.gamma = gamma
        self.bandwidth_bit = bandwidth_bit
        with open('model/gac.cu', 'r') as file:
            code = file.read()
            self.cu = cp.RawKernel(code, 'gac')
            self.lut = cp.RawKernel(code, 'slut')
    def clipping(self):
        cp.clip(self.img, 0, self.clip, out=self.img)
        return self.img
    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        img_c = self.img.shape[2]
        
        gc_img = cp.empty((img_h, img_w, img_c), cp.uint16)
        #print(self.lut)
        #nlm_img = cp.zeros((raw_h, raw_w), cp.int16)       
        bw = self.bandwidth_bit
        #mode = 'rgb'

        maxval = pow(2,bw)
        lut = cp.zeros(maxval,cp.uint16)
        divide = pow(2,int((bw-8)))
        #ind = range(0, maxval)
        #for i in ind :
        #    lut[i] = round(pow(float(i)/maxval, gamma) * maxval)
            
        self.lut((maxval,1),(64,1),(lut,self.gamma,maxval))
        #for i in ind :
        #print(lut)
        self.cu((img_w//32,img_h//24), (32,24), (self.img,img_w,img_h,lut,divide,maxval ,gc_img))  # grid, block and arguments  
        '''
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if self.mode == 'rgb':
                    gc_img[y, x, 0] = self.lut[self.img[y, x, 0]]
                    gc_img[y, x, 1] = self.lut[self.img[y, x, 1]]
                    gc_img[y, x, 2] = self.lut[self.img[y, x, 2]]
                    gc_img[y, x, :] = gc_img[y, x, :] / 8
                elif self.mode == 'yuv':
                    gc_img[y, x, 0] = self.lut[0][self.img[y, x, 0]]
                    gc_img[y, x, 1] = self.lut[1][self.img[y, x, 1]]
                    gc_img[y, x, 2] = self.lut[1][self.img[y, x, 2]]
        
        '''
        self.img = gc_img
        return self.img
