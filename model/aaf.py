#!/usr/bin/python
import numpy as np
from scipy.ndimage import correlate
import cupyx.scipy.ndimage as cnd
import cupy as cp
class AAF:
    'Anti-aliasing Filter'

    def __init__(self, img):
        self.img = img
        with open('model/aaf.cu', 'r') as file:
            code = file.read()
            self.cu = cp.RawKernel(code, 'aaf')
    def padding(self):
        img_pad = cp.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def execute(self):
        img_pad = self.padding()
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        img_h = img_pad.shape[0]
        img_w = img_pad.shape[1]
        #aaf_img = cp.empty((raw_h, raw_w), cp.uint16)
        filter = cp.array([ [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 8, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1]])/16        
        '''
        aaf_img = cnd.correlate(self.img, cp.array([[1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 8, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])/16)
        '''
        #print(img_w,img_h,img_w-raw_w,img_h-raw_h,filter)

        self.cu((img_w//32,img_h//24), (32,24), (img_pad,raw_w,raw_h,img_w-raw_w,img_h-raw_h,filter ,self.img))  # grid, block and arguments  
        #self.img = aaf_img
        return self.img

