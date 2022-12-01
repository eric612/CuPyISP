#!/usr/bin/python
import numpy as np
import cupy as cp

import cv2 
class CNF:
    'Chroma Noise Filtering'

    def __init__(self, img, bayer_pattern, thres, gain, clip):
        self.img = img
        self.bayer_pattern = bayer_pattern
        self.thres = thres
        self.gain = gain
        self.clip = clip

    def padding(self):
        img_pad = cp.pad(self.img, ((4, 4), (4, 4)), 'reflect')
        return img_pad

    def clipping(self):
        cp.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def cnc(self, is_color, center, avgG, avgC1, avgC2):
  
        'Chroma Noise Correction'
        r_gain = self.gain[0]
        gr_gain = self.gain[1]
        gb_gain = self.gain[2]
        b_gain = self.gain[3]
        dampFactor = 1.0
        signalGap = center - max(avgG, avgC2)
        if is_color == 'r':
            if r_gain <= 1.0:
                dampFactor = 1.0
            elif r_gain > 1.0 and r_gain <= 1.2:
                dampFactor = 0.5
            elif r_gain > 1.2:
                dampFactor = 0.3
        elif is_color == 'b':
            if b_gain <= 1.0:
                dampFactor = 1.0
            elif b_gain > 1.0 and b_gain <= 1.2:
                dampFactor = 0.5
            elif b_gain > 1.2:
                dampFactor = 0.3
        chromaCorrected = max(avgG, avgC2) + dampFactor * signalGap
        if is_color == 'r':
            signalMeter = 0.299 * avgC1 + 0.587 * avgG + 0.114 * avgC2
        elif is_color == 'b':
            signalMeter = 0.299 * avgC2 + 0.587 * avgG + 0.114 * avgC1
        if signalMeter <= 30:
            fade1 = 1.0
        elif signalMeter > 30 and signalMeter <= 50:
            fade1 = 0.9
        elif signalMeter > 50 and signalMeter <= 70:
            fade1 = 0.8
        elif signalMeter > 70 and signalMeter <= 100:
            fade1 = 0.7
        elif signalMeter > 100 and signalMeter <= 150:
            fade1 = 0.6
        elif signalMeter > 150 and signalMeter <= 200:
            fade1 = 0.3
        elif signalMeter > 200 and signalMeter <= 250:
            fade1 = 0.1
        else:
            fade1 = 0
        if avgC1 <= 30:
            fade2 = 1.0
        elif avgC1 > 30 and avgC1 <= 50:
            fade2 = 0.9
        elif avgC1 > 50 and avgC1 <= 70:
            fade2 = 0.8
        elif avgC1 > 70 and avgC1 <= 100:
            fade2 = 0.6
        elif avgC1 > 100 and avgC1 <= 150:
            fade2 = 0.5
        elif avgC1 > 150 and avgC1 <= 200:
            fade2 = 0.3
        elif avgC1 > 200:
            fade2 = 0
        fadeTot = fade1 * fade2
        center_out = (1 - fadeTot) * center + fadeTot * chromaCorrected
        return center_out

    def cnd(self, y, x, img):
    

        'Chroma Noise Detection'
        avgG = 0
        avgC1 = 0
        avgC2 = 0
        is_noise = 0
        for i in range(y - 4, y + 4, 1):
            for j in range(x - 4, x + 4, 1):
                if i % 2 == 1 and j % 2 == 0:
                    avgG = avgG + img[i,j]
                elif i % 2 == 0 and j % 2 == 1:
                    avgG = avgG + img[i, j]
                elif i % 2 == 0 and j % 2 == 0:
                    avgC1 = avgC1 + img[i,j]    # weights are equal, could be as gaussian dist
                elif i % 2 == 1 and j % 2 == 1:
                    avgC2 = avgC2 + img[i,j]
        avgG = avgG / 40
        avgC1 = avgC1 / 25
        avgC2 = avgC2 / 16
        center = img[y, x]
        if center > avgG + self.thres and center > avgC2 + self.thres:
            if avgC1 > avgG + self.thres and avgC1 > avgC2 + self.thres:
                is_noise = 1
            else:
                is_noise = 0
        else:
            is_noise = 0
        return is_noise, avgG, avgC1, avgC2
    def cnd_img(self,img,img_out):
        with open('model/cnf.cu', 'r') as file:
            code = file.read()
        cnf = cp.RawKernel(code, 'cnf')
        raw_h = img.shape[0]
        raw_w = img.shape[1]
        pad_h = raw_h - img_out.shape[0]
        pad_w = raw_w - img_out.shape[1]

        if self.bayer_pattern == 'rggb':
            type = 0
        elif self.bayer_pattern == 'bggr':
            type = 1
        elif self.bayer_pattern == 'gbrg':
            type = 2
        elif self.bayer_pattern == 'grbg':
            type = 3
        #print(pad_w,pad_h)
        cnf((raw_w//56,raw_h//56), (28,28), (img,raw_w,raw_h,pad_w,pad_h,4,4,self.thres,self.gain[0],self.gain[3],type,img_out))  # grid, block and arguments    
        #plt.imshow(is_noise_img.get(), cmap='gray')
        #plt.show()
    def cnf(self, is_color, y, x, img):

        is_noise, avgG, avgC1, avgC2 = self.cnd(y, x, img)
        if is_noise:
            pix_out = self.cnc(is_color, img[y,x], avgG, avgC1, avgC2)
        else:
            pix_out = img[y,x]
        return pix_out
    def copy(self,img,img_out): 
        raw_h = img_out.shape[0]
        raw_w = img_out.shape[1]
        pad_h = img.shape[0]-raw_h
        pad_w = img.shape[1]-raw_w
        print(pad_w)
        copy = cp.RawKernel(r'''
        extern "C" __global__
        void copy(const unsigned short* p, unsigned short* z,int width, int height,int pad_w,int pad_h) {

            int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
            int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
            int i_width = width+pad_w;
            if ((row < height) && (col < width)) {
                int offset = (row+4)*i_width + col + 4;
                int shift_offset = (row)*width + col ;
                z[shift_offset] = p[offset];
                z[shift_offset+1] = p[offset+1];
                z[shift_offset+width] = p[offset+i_width];
                z[shift_offset+width+1] = p[offset+i_width+1];
            }
         }
        ''', 'copy')


        #Invoking the kernel with a grid of 30 blocks, each consisting of 24 threads
        copy((raw_w//8,raw_h//8), (4,4), (img, img_out,raw_w,raw_h,pad_w,pad_h))  # grid, block and arguments
        
    def execute(self):
        img_pad = self.padding()
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        #print(img_pad.shape,',self.bayer_pattern:',self.bayer_pattern)
        cnf_img = cp.zeros((raw_h, raw_w), dtype=cp.uint16)
        #print(cnf_img.dtype)
        #Defining the CUDA kernel
        #print(img_pad.shape)
        #self.copy(img_pad,cnf_img)
        self.cnd_img(img_pad,cnf_img)
        #cv2.imwrite('output.png', cnf_img.get())
        #plt.imshow(self.img.get(), cmap='gray')
        #plt.show()
        '''
        for y in range(0, img_pad.shape[0] - 8 - 1, 2):
            for x in range(0, img_pad.shape[1] - 8 - 1, 2):
                if self.bayer_pattern == 'rggb':
                    r = img_pad[y + 4, x + 4]
                    gr = img_pad[y + 4, x + 5]
                    gb = img_pad[y + 5, x + 4]
                    b = img_pad[y + 5, x + 5]
                    cnf_img[y, x] = self.cnf('r', y + 4, x + 4, img_pad)
                    cnf_img[y, x + 1] = gr
                    cnf_img[y + 1, x] = gb
                    cnf_img[y + 1, x + 1] = self.cnf('b', y + 5, x + 5, img_pad)
                elif self.bayer_pattern == 'bggr':
                    b = img_pad[y + 4, x + 4]
                    gb = img_pad[y + 4, x + 5]
                    gr = img_pad[y + 5, x + 4]
                    r = img_pad[y + 5, x + 5]
                    cnf_img[y, x] = self.cnf('b', y + 4, x + 4, img_pad)
                    cnf_img[y, x + 1] = gb
                    cnf_img[y + 1, x] = gr
                    cnf_img[y + 1, x + 1] = self.cnf('r', y + 5, x + 5, img_pad)
                elif self.bayer_pattern == 'gbrg':
                    gb = img_pad[y + 4, x + 4]
                    b = img_pad[y + 4, x + 5]
                    r = img_pad[y + 5, x + 4]
                    gr = img_pad[y + 5, x + 5]
                    cnf_img[y, x] = gb
                    cnf_img[y, x + 1] = self.cnf('b', y + 4, x + 5, img_pad)
                    cnf_img[y + 1, x] = self.cnf('r', y + 5, x + 4, img_pad)
                    cnf_img[y + 1, x + 1] = gr
                elif self.bayer_pattern == 'grbg':
                    gr = img_pad[y + 4, x + 4]
                    r = img_pad[y + 4, x + 5]
                    b = img_pad[y + 5, x + 2]
                    gb = img_pad[y + 5, x + 5]
                    cnf_img[y, x, :] = gr
                    cnf_img[y, x + 1, :] = self.cnf('r', y + 4, x + 5, img_pad)
                    cnf_img[y + 1, x, :] = self.cnf('b', y + 5, x + 4, img_pad)
                    cnf_img[y + 1, x + 1, :] = gb
        '''
        self.img = cnf_img
        return self.clipping()
