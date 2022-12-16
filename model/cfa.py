#!/usr/bin/python
import numpy as np
import cupy as cp
import cv2 
class CFA:
    'Color Filter Array Interpolation'

    def __init__(self, img, mode, bayer_pattern, clip):
        self.img = img
        self.mode = mode
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def padding(self):
        img_pad = cp.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        return img_pad

    def clipping(self):
        cp.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def malvar(self, is_color, center, y, x, img):
        if is_color == 'r':
            r = center
            g = 4 * img[y,x] - img[y-2,x] - img[y,x-2] - img[y+2,x] - img[y,x+2] \
                + 2 * (img[y+1,x] + img[y,x+1] + img[y-1,x] + img[y,x-1])
            b = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
                + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
            g = g / 8
            b = b / 8
        elif is_color == 'gr':
            r = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
                + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
            g = center
            b = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
                + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
            r = r / 8
            b = b / 8
        elif is_color == 'gb':
            r = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
                + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
            g = center
            b = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
                + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
            r = r / 8
            b = b / 8
        elif is_color == 'b':
            r = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
                + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
            g = 4 * img[y,x] - img[y-2,x] - img[y,x-2] - img[y+2,x] - img[y,x+2] \
                + 2 * (img[y+1,x] + img[y,x+1] + img[y-1,x] + img[y,x-1])
            b = center
            r = r / 8
            g = g / 8
        return [r, g, b]

    def execute(self):
        img_pad = self.padding()
        img_pad = img_pad.astype(cp.int16)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        #print(self.img.shape)
        cfa_img = cp.empty((raw_h, raw_w, 3), cp.int16)
        pre_map = cp.empty((raw_h, raw_w), cp.int16)
        with open('model/cfa.cu', 'r') as file:
            code = file.read()
        cfa = cp.RawKernel(code, 'cfa')
        pre_maps = cp.RawKernel(code, 'pre_maps')
        
        if self.bayer_pattern == 'rggb':
            type = 0
        elif self.bayer_pattern == 'bggr':
            type = 1
        elif self.bayer_pattern == 'gbrg':
            type = 2
        elif self.bayer_pattern == 'grbg':
            type = 3
        elif self.bayer_pattern == 'rccc':
            type = 4
        #print(type)
        pad_w = img_pad.shape[1] - raw_w
        pad_h = img_pad.shape[0] - raw_h
        pre_maps((raw_w//32,raw_h//24), (16,12), (img_pad,raw_w,raw_h,pad_w,pad_h,type,pre_map))  # grid, block and arguments  
        
        done = pre_map.get()/4096
        #cv2.imshow('cv', done)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()       
        #cv2.imwrite('cfa.jpg', done*256)
        
        cfa((raw_w//32,raw_h//24), (16,12), (img_pad,pre_map,raw_w,raw_h,pad_w,pad_h,type,cfa_img))  # grid, block and arguments
        
        #done = cfa_img.get()/4096
        #cv2.imshow('cv', done)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite('cfa.jpg', done*256)
        
        '''
        for y in range(0, img_pad.shape[0]-4-1, 2):
            for x in range(0, img_pad.shape[1]-4-1, 2):
                if self.bayer_pattern == 'rggb':
                    r = img_pad[y+2,x+2]
                    gr = img_pad[y+2,x+3]
                    gb = img_pad[y+3,x+2]
                    b = img_pad[y+3,x+3]
                    if self.mode == 'malvar':
                        cfa_img[y,x,:] = self.malvar('r', r, y+2,x+2, img_pad)
                        cfa_img[y,x+1,:] = self.malvar('gr', gr, y+2,x+3, img_pad)
                        cfa_img[y+1,x,:] = self.malvar('gb', gb, y+3,x+2, img_pad)
                        cfa_img[y+1,x+1,:] = self.malvar('b', b, y+3,x+3, img_pad)
                elif self.bayer_pattern == 'bggr':
                    b = img_pad[y+2,x+2]
                    gb = img_pad[y+2,x+3]
                    gr = img_pad[y+3,x+2]
                    r = img_pad[y+3,x+3]
                    if self.mode == 'malvar':
                        cfa_img[y,x,:] = self.malvar('b', b, y+2,x+2, img_pad)
                        cfa_img[y,x+1,:] = self.malvar('gb', gb, y+2,x+3, img_pad)
                        cfa_img[y+1,x,:] = self.malvar('gr', gr, y+3,x+2, img_pad)
                        cfa_img[y+1,x+1,:] = self.malvar('r', r, y+3,x+3, img_pad)
                elif self.bayer_pattern == 'gbrg':
                    gb = img_pad[y+2,x+2]
                    b = img_pad[y+2,x+3]
                    r = img_pad[y+3,x+2]
                    gr = img_pad[y+3,x+3]
                    if self.mode == 'malvar':
                        cfa_img[y,x,:] = self.malvar('gb', gb, y+2,x+2, img_pad)
                        cfa_img[y,x+1,:] = self.malvar('b', b, y+2,x+3, img_pad)
                        cfa_img[y+1,x,:] = self.malvar('r', r, y+3,x+2, img_pad)
                        cfa_img[y+1,x+1,:] = self.malvar('gr', gr, y+3,x+3, img_pad)
                elif self.bayer_pattern == 'grbg':
                    gr = img_pad[y+2,x+2]
                    r = img_pad[y+2,x+3]
                    b = img_pad[y+3,x+2]
                    gb = img_pad[y+3,x+3]
                    if self.mode == 'malvar':
                        cfa_img[y,x,:] = self.malvar('gr', gr, y+2,x+2, img_pad)
                        cfa_img[y,x+1,:] = self.malvar('r', r, y+2,x+3, img_pad)
                        cfa_img[y+1,x,:] = self.malvar('b', b, y+3,x+2, img_pad)
                        cfa_img[y+1,x+1,:] = self.malvar('gb', gb, y+3,x+3, img_pad)
        '''
        #cfa_img[...,0] = self.img
        #cfa_img[...,1] = self.img
        #cfa_img[...,2] = self.img
        self.img = cfa_img
        return self.clipping()