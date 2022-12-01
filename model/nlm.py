#!/usr/bin/python
import numpy as np
import cupy as cp

class NLM:
    'Non-Local Means Denoising'

    def __init__(self, img, ds, Ds, h, clip):
        self.img = img
        self.ds = ds    # neighbour window size - 1 /2
        self.Ds = Ds    # search window size - 1 / 2
        self.h = h
        self.clip = clip
        #print(ds,Ds,h,clip)
        with open('model/nlm.cu', 'r') as file:
            code = file.read()
            self.cu = cp.RawKernel(code, 'nlm')
    def padding(self):
        img_pad = cp.pad(self.img, (self.Ds, self.Ds), 'reflect')
        return img_pad

    def clipping(self):
        cp.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def calWeights(self, img, kernel, y, x):
        wmax = 0
        sweight = 0
        average = 0
        for j in range(2 * self.Ds + 1 - 2 * self.ds - 1):
            for i in range(2 * self.Ds + 1 - 2 * self.ds - 1):
                start_y = y - self.Ds + self.ds + j
                start_x = x - self.Ds + self.ds + i
                neighbour_w = img[start_y - self.ds:start_y + self.ds + 1, start_x - self.ds:start_x + self.ds + 1]
                
                center_w = img[y-self.ds:y+self.ds+1, x-self.ds:x+self.ds+1]
                
                if j != y or i != x:
                    sub = np.subtract(neighbour_w, center_w)
                    
                    dist = np.sum(np.multiply(kernel, np.multiply(sub, sub)))
                    w = np.exp(-dist/pow(self.h, 2))    # replaced by look up table
                    #print(pow(self.h, 2),-dist/pow(self.h, 2),w)
                    if w > wmax:
                        wmax = w
                    sweight = sweight + w
                    average = average + w * img[start_y, start_x]
        return sweight, average, wmax

    def execute(self):
        img_pad = self.padding()
        img_pad = img_pad.astype(cp.uint16)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        pad_w = img_pad.shape[1] - raw_w
        pad_h = img_pad.shape[0] - raw_h
        #print(pad_w,pad_h)
        nlm_img = cp.zeros((raw_h, raw_w), cp.int16)
        kernel = cp.ones((2*self.ds+1, 2*self.ds+1)) / pow(2*self.ds+1, 2)
        #print(2*self.ds+1)
        #for j in range(2 * self.Ds + 1 - 2 * self.ds - 1):
        #    print(j)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        self.cu((raw_w//32,raw_h//24), (32,24), (img_pad,raw_w,raw_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments  
        
        #print(kernel,self.Ds,self.ds)
        '''
        for y in range(img_pad.shape[0] - 2 * self.Ds):
            for x in range(img_pad.shape[1] - 2 * self.Ds):
                center_y = y + self.Ds
                center_x = x + self.Ds
                sweight, average, wmax = self.calWeights(img_pad, kernel, center_y, center_x)
                average = average + wmax * img_pad[center_y, center_x]
                sweight = sweight + wmax
                nlm_img[y,x] = average / sweight
        '''
        self.img = nlm_img.astype(cp.uint8)
        
        return self.clipping()

