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
    bayer = cp.zeros((raw_h, raw_w), dtype=cp.uint8)
    convert((raw_w//30,raw_h//24), (15,12),(img_gpu,raw_w,raw_h,0,bayer))
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

