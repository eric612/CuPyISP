import cv2 
import cupy as cp
import yaml


def clipping(img):
    cp.clip(img, 0, 255, out=img)
    return img
def RGB_to_Bayer(path='kodim19.png') :
    with open('tool/convert.cu', 'r') as file:
        code = file.read()
    convert = cp.RawKernel(code, 'convert')
            
    # 讀取圖檔
    img = cv2.imread(path)
    img_gpu = cp.asarray(img)
    raw_h = img.shape[0]
    raw_w = img.shape[1]
    bayer = cp.zeros((raw_h, raw_w), dtype=cp.uint16)
    convert((raw_w//30,raw_h//24), (15,12),(img_gpu,raw_w,raw_h,4,bayer))
    return bayer
    #print(img.dtype)

    #awb_img = cp.empty((raw_h, raw_w))
    #awb_img = img;
    #awb_img[:,:,1] = clipping(img[:,:,1]*2)
    # 顯示圖片
    #cv2.imshow('My Image', bayer.get())

    # 按下任意鍵則關閉所有視窗
    #cv2.waitKey(3000)
    #cv2.destroyAllWindows()

def C_R(path='kodim19.png') :
    with open('tool/convert.cu', 'r') as file:
        code = file.read()
    convert = cp.RawKernel(code, 'convert_C_R')
            
    # 讀取圖檔
    img = cv2.imread(path)
    img_gpu = cp.asarray(img)
    raw_h = img.shape[0]
    raw_w = img.shape[1]
    bayer = cp.zeros((raw_h, raw_w), dtype=cp.uint8)
    convert((raw_w//32,raw_h//24), (32,24),(img_gpu,raw_w,raw_h,bayer))
    return bayer
def C_R_Raw(path='ov10642_With_Noise.raw',bit=12) :
    with open('tool/convert.cu', 'r') as file:
        code = file.read()
    convert = cp.RawKernel(code, 'Analyze_Raw')
            
    # 讀取圖檔
    raw_w = 1280
    raw_h = 720

    rawimg = cp.fromfile(path, dtype='uint16', sep='')
    rawimg = rawimg.reshape([raw_h, raw_w])
    max = 1 << bit
    CR = cp.zeros((raw_h, raw_w), dtype=cp.uint16)
    convert((raw_w//32,raw_h//24), (16,12),(rawimg,raw_w,raw_h,CR))
    return CR/max
def C_R_RawRecover(path='ov10642_With_Noise.raw',bit=12) :
    with open('tool/convert.cu', 'r') as file:
        code = file.read()
    convert = cp.RawKernel(code, 'Recover_Raw')
            
    # 讀取圖檔
    raw_w = 1280
    raw_h = 720

    rawimg = cp.fromfile(path, dtype='uint16', sep='')
    rawimg = rawimg.reshape([raw_h, raw_w])
    max = 1 << bit
    CR = cp.zeros((raw_h, raw_w,3), dtype=cp.uint16)
    convert((raw_w//32,raw_h//24), (16,12),(rawimg,raw_w,raw_h,max,CR))
    return CR*4/max       
if __name__ == '__main__':

    #img = C_R()
    img = C_R_RawRecover()
    cv2.imshow('cv', img.get())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('C_R.jpg', img.get()*255)