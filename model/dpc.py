#!/usr/bin/python
import numpy as np
import cupy as cp
class DPC:
    'Dead Pixel Correction'

    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = int(thres)
        self.mode = mode
        self.clip = clip
        cu = 'model/dpc.cu'
        with open(cu, 'r') as file:
            code = file.read()
        self.kernel = cp.RawKernel(code, 'dpc')
        self.proc = 'gpu'

    def padding(self):
        if self.proc == 'gpu':
            img_pad = cp.pad(self.img, (2, 2), 'reflect')
        else:
            img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def clipping(self):
        if self.proc == 'gpu':
            cp.clip(self.img, 0, self.clip, out=self.img)
            return self.img
        else:
            np.clip(self.img, 0, self.clip, out=self.img)
            return self.img

    def execute(self):
        img_pad = self.padding()
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        pad_h = 2
        pad_w = 2
        #print(self.mode)
        if self.proc == 'gpu':
            dpc_img = cp.empty((raw_h, raw_w), np.uint16)
            self.kernel((raw_w//32,raw_h//24), (32,24), (img_pad,dpc_img,raw_w,raw_h,pad_w,pad_h,int(self.thres)) )
        else :
            dpc_img = np.empty((raw_h, raw_w), np.uint16)
            for y in range(img_pad.shape[0] - 4):
                for x in range(img_pad.shape[1] - 4):
                    p0 = int(img_pad[y + 2, x + 2])
                    p1 = int(img_pad[y, x])
                    p2 = int(img_pad[y, x + 2])
                    p3 = int(img_pad[y, x + 4])
                    p4 = int(img_pad[y + 2, x])
                    p5 = int(img_pad[y + 2, x + 4])
                    p6 = int(img_pad[y + 4, x])
                    p7 = int(img_pad[y + 4, x + 2])
                    p8 = int(img_pad[y + 4, x + 4])
                    if (abs(p1 - p0) > self.thres) and (abs(p2 - p0) > self.thres) and (abs(p3 - p0) > self.thres) \
                            and (abs(p4 - p0) > self.thres) and (abs(p5 - p0) > self.thres) and (abs(p6 - p0) > self.thres) \
                            and (abs(p7 - p0) > self.thres) and (abs(p8 - p0) > self.thres):
                        if self.mode == 'mean':
                            p0 = (p2 + p4 + p5 + p7) / 4
                        elif self.mode == 'gradient':
                            dv = abs(2 * p0 - p2 - p7)
                            dh = abs(2 * p0 - p4 - p5)
                            ddl = abs(2 * p0 - p1 - p8)
                            ddr = abs(2 * p0 - p3 - p6)
                            if (min(dv, dh, ddl, ddr) == dv):
                                p0 = (p2 + p7 + 1) / 2
                            elif (min(dv, dh, ddl, ddr) == dh):
                                p0 = (p4 + p5 + 1) / 2
                            elif (min(dv, dh, ddl, ddr) == ddl):
                                p0 = (p1 + p8 + 1) / 2
                            else:
                                p0 = (p3 + p6 + 1) / 2
                    dpc_img[y, x] = p0
        self.img = dpc_img
        return self.clipping()

