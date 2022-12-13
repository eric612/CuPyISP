#!/usr/bin/python
import numpy as np
import cupy as cp
from model.nlm import NLM
class DWT:
    'Discrete Wavelet Transform'

    def __init__(self, img):
        self.img = img
        self.ds = 1    # neighbour window size - 1 /2
        self.Ds = 3    # search window size - 1 / 2
        self.h = 5
        self.clip = 255
        with open('model/dwt.cu', 'r') as file:
            code = file.read()
            self.cu_dwt_h = cp.RawKernel(code, 'dwt_h')
            self.cu_dwt_v = cp.RawKernel(code, 'dwt_v')
            self.cu_idwt_h = cp.RawKernel(code, 'idwt_h')
            self.cu_idwt_v = cp.RawKernel(code, 'idwt_v')

        with open('model/nlm.cu', 'r') as file:
            code = file.read()
            self.denoise = cp.RawKernel(code, 'nlm')
    def clipping(self,img):
        cp.clip(img, 0, self.clip, out=img)
        return img
    def padding(self,img):
        img_pad = cp.pad(img, (self.Ds, self.Ds), 'reflect')
        return img_pad
    def luminance_denoise(self,image):
        img_h = image.shape[0]
        img_w = image.shape[1]
        #self.img = self.img.astype(cp.uint8)
        img = cp.asarray(image, dtype='int16')
        
        dwt_img = cp.empty((img_h, img_w), cp.int16)

        self.cu_dwt_h((img_w//32,img_h//24), (16,24), (img,img_w,img_h,dwt_img))  # grid, block and arguments
        self.cu_dwt_v((img_w//32,img_h//24), (32,12), (dwt_img,img_w,img_h,img))  # grid, block and arguments
        
        img_w = img_w//2
        img_h = img_h//2
        img2_L1 = img[:img_h,:img_w].copy()
        dwt_img_L1 = cp.empty((img_h, img_w), cp.int16)
        
        self.cu_dwt_h((img_w//8,img_h//8), (4,8), (img2_L1,img_w,img_h,dwt_img_L1))  # grid, block and arguments
        self.cu_dwt_v((img_w//8,img_h//8), (8,4), (dwt_img_L1,img_w,img_h,img2_L1))  # grid, block and arguments
        
        img_w = img_w//2
        img_h = img_h//2
        img2_L2 = img2_L1[:img_h,:img_w].copy()
        
        
        
        dwt_img_L2 = cp.empty((img_h, img_w), cp.int16)
        self.cu_dwt_h((img_w//4,img_h//4), (2,4), (img2_L2,img_w,img_h,dwt_img_L2))  # grid, block and arguments
        self.cu_dwt_v((img_w//4,img_h//4), (4,2), (dwt_img_L2,img_w,img_h,img2_L2))  # grid, block and arguments
        '''
        img_w = img_w//2
        img_h = img_h//2
        img2_L3 = img2_L2[:img_h,:img_w].copy()
        
        # non-local means denoising 1/8 scale

        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        img_pad = self.padding(img2_L3)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        #print(img_w,img_h)
        self.denoise((img_w//16,img_h//9), (16,9), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments  
        
        img2_L2[:img_h,:img_w] = nlm_img
        img_w = img_w*2
        img_h = img_h*2  
        '''
        self.cu_idwt_v((img_w//4,img_h//4), (4,2), (img2_L2,img_w,img_h,dwt_img_L2))  # grid, block and arguments
        self.cu_idwt_h((img_w//4,img_h//4), (2,4), (dwt_img_L2,img_w,img_h,img2_L2))  # grid, block and arguments
        
      
        
        # non-local means denoising 1/4 scale

        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        img_pad = self.padding(img2_L2)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        #print(pad_w,pad_h)
        #print(img_w,img_h)
        self.denoise((img_w//16,img_h//9), (16,9), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments  
        
        
        img2_L1[:img_h,:img_w] = nlm_img
        img_w = img_w*2
        img_h = img_h*2
        self.cu_idwt_v((img_w//8,img_h//8), (8,4), (img2_L1,img_w,img_h,dwt_img_L1))  # grid, block and arguments
        self.cu_idwt_h((img_w//8,img_h//8), (4,8), (dwt_img_L1,img_w,img_h,img2_L1))  # grid, block and arguments
        
        # non-local means denoising 1/2 scale
        
        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        img_pad = self.padding(img2_L1)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        
        self.denoise((img_w//32,img_h//18), (32,18), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments   
        
        
        img[:img_h,:img_w] = nlm_img
        img_w = img_w*2
        img_h = img_h*2   
        
        self.cu_idwt_v((img_w//32,img_h//24), (32,12), (img,img_w,img_h,dwt_img))  # grid, block and arguments
        self.cu_idwt_h((img_w//32,img_h//24), (16,24), (dwt_img,img_w,img_h,img))  # grid, block and arguments
        
        # non-local means denoising 1/1 scale

        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        
        img_pad = self.padding(img)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        #print(pad_w,pad_h)
        
        self.denoise((img_w//32,img_h//24), (32,24), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,img))  # grid, block and arguments            
        #self.img = dwt_img
        
        #self.img = cp.absolute(img2_L1).astype(cp.uint16)
        #self.img = cp.asarray(img2_L2, dtype='uint8')      
        return img
    def chroma_denoise(self,image):
        self.ds = 1    # neighbour window size - 1 /2
        self.Ds = 4    # search window size - 1 / 2
        self.h = 5
        
        img_h = image.shape[0]
        img_w = image.shape[1]
        #self.img = self.img.astype(cp.uint8)
        img = cp.asarray(image, dtype='int16')
        
        dwt_img = cp.empty((img_h, img_w), cp.int16)

        self.cu_dwt_h((img_w//32,img_h//24), (16,24), (img,img_w,img_h,dwt_img))  # grid, block and arguments
        self.cu_dwt_v((img_w//32,img_h//24), (32,12), (dwt_img,img_w,img_h,img))  # grid, block and arguments
        
        img_w = img_w//2
        img_h = img_h//2
        img2_L1 = img[:img_h,:img_w].copy()
        dwt_img_L1 = cp.empty((img_h, img_w), cp.int16)
        
        self.cu_dwt_h((img_w//8,img_h//8), (4,8), (img2_L1,img_w,img_h,dwt_img_L1))  # grid, block and arguments
        self.cu_dwt_v((img_w//8,img_h//8), (8,4), (dwt_img_L1,img_w,img_h,img2_L1))  # grid, block and arguments
        
        img_w = img_w//2
        img_h = img_h//2
        img2_L2 = img2_L1[:img_h,:img_w].copy()
              
        
        # non-local means denoising 1/4 scale

        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        img_pad = self.padding(img2_L2)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        #print(pad_w,pad_h)
        #print(img_w,img_h)
        self.denoise((img_w//32,img_h//18), (32,18), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments  
        
        
        img2_L1[:img_h,:img_w] = nlm_img
        img_w = img_w*2
        img_h = img_h*2
        self.cu_idwt_v((img_w//8,img_h//8), (8,4), (img2_L1,img_w,img_h,dwt_img_L1))  # grid, block and arguments
        self.cu_idwt_h((img_w//8,img_h//8), (4,8), (dwt_img_L1,img_w,img_h,img2_L1))  # grid, block and arguments
        
        # non-local means denoising 1/2 scale
        
        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        img_pad = self.padding(img2_L1)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        
        self.denoise((img_w//32,img_h//18), (32,18), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,nlm_img))  # grid, block and arguments   
        
        
        img[:img_h,:img_w] = nlm_img
        img_w = img_w*2
        img_h = img_h*2   
        
        self.cu_idwt_v((img_w//32,img_h//24), (32,12), (img,img_w,img_h,dwt_img))  # grid, block and arguments
        self.cu_idwt_h((img_w//32,img_h//24), (16,24), (dwt_img,img_w,img_h,img))  # grid, block and arguments
        
        # non-local means denoising 1/1 scale

        nlm_img = cp.zeros((img_h, img_w), cp.int16)
        
        img_pad = self.padding(img)
        search_dis = (self.Ds-1)*2
        kernel_size = 2*self.ds+1
        pad_w = img_pad.shape[1] - img_w
        pad_h = img_pad.shape[0] - img_h
        #print(pad_w,pad_h)
        
        self.denoise((img_w//32,img_h//24), (32,24), (img_pad,img_w,img_h,pad_w,pad_h,search_dis,kernel_size,self.h,img))  # grid, block and arguments            
        #self.img = dwt_img
        
        #img = cp.absolute(img2_L1).astype(cp.uint16)
        #self.img = cp.asarray(img2_L2, dtype='uint8')      
        return img        
    def execute(self):
        img = self.luminance_denoise(self.img[...,0])
        self.img[...,0] = (img).astype(cp.uint8)
        img = self.chroma_denoise(self.img[...,1])
        self.img[...,1] = (img).astype(cp.uint8)
        img = self.chroma_denoise(self.img[...,2])
        self.img[...,2] = (img).astype(cp.uint8)
        #self.img = (img).astype(cp.uint8)
        return self.img
