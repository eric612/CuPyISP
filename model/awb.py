#!/usr/bin/python
import numpy as np
import cupy as cp
class WBGC:
    'Auto White Balance Gain Control'

    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        cp.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        r_gain = self.parameter[0]
        gr_gain = self.parameter[1]
        gb_gain = self.parameter[2]
        b_gain = self.parameter[3]
        #print(self.parameter)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        awb_img = cp.empty((raw_h, raw_w), cp.int16)
        if self.bayer_pattern == 'rggb':
            r = self.img[::2, ::2] * r_gain
            b = self.img[1::2, 1::2] * b_gain
            gr = self.img[::2, 1::2] * gr_gain
            gb = self.img[1::2, ::2] * gb_gain
            awb_img[::2, ::2] = r
            awb_img[::2, 1::2] = gr
            awb_img[1::2, ::2] = gb
            awb_img[1::2, 1::2] = b
        elif self.bayer_pattern == 'bggr':
            b = self.img[::2, ::2] * b_gain
            r = self.img[1::2, 1::2] * r_gain
            gb = self.img[::2, 1::2] * gb_gain
            gr = self.img[1::2, ::2] * gr_gain
            awb_img[::2, ::2] = b
            awb_img[::2, 1::2] = gb
            awb_img[1::2, ::2] = gr
            awb_img[1::2, 1::2] = r
        elif self.bayer_pattern == 'gbrg':
            b = self.img[::2, 1::2] * b_gain
            r = self.img[1::2, ::2] * r_gain
            gb = self.img[::2, ::2] * gb_gain
            gr = self.img[1::2, 1::2] * gr_gain
            awb_img[::2, ::2] = gb
            awb_img[::2, 1::2] = b
            awb_img[1::2, ::2] = r
            awb_img[1::2, 1::2] = gr
        elif self.bayer_pattern == 'grbg':
            r = self.img[::2, 1::2] * r_gain
            b = self.img[1::2, ::2] * b_gain
            gr = self.img[::2, ::2] * gr_gain
            gb = self.img[1::2, 1::2] * gb_gain
            awb_img[::2, ::2] = gr
            awb_img[::2, 1::2] = r
            awb_img[1::2, ::2] = b
            awb_img[1::2, 1::2] = gb
        self.img = awb_img
        return self.clipping()
