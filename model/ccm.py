#!/usr/bin/python
import numpy as np
import cupy as cp
class CCM:
    'Color Correction Matrix'

    def __init__(self, img, ccm):
        self.img = img
        self.ccm = ccm

    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        img_c = self.img.shape[2]
        ccm_img = cp.empty((img_h, img_w, img_c), cp.uint32)
        for y in range(img_h):
            for x in range(img_w):
                mulval = self.ccm[:,0:3] * self.img[y,x,:]
                ccm_img[y,x,0] = cp.sum(mulval[0]) + self.ccm[0,3]
                ccm_img[y,x,1] = cp.sum(mulval[1]) + self.ccm[1,3]
                ccm_img[y,x,2] = cp.sum(mulval[2]) + self.ccm[2,3]
                ccm_img[y,x,:] = ccm_img[y,x,:] / 1024
        self.img = ccm_img.astype(cp.uint8)
        return self.img
