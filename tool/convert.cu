
#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )

extern "C" __global__
void convert(const unsigned char* img,int width, int height,int bayer_pattern,unsigned short* img_out) {
    //convert RGB to Bayer
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    if ((row < height) && (col < width)) {
        int offset_bayer = (row)*width + col;
        int offset_rgb = (row)*width*3 + col*3;
        if (bayer_pattern==0) {
            img_out[offset_bayer] = img[offset_rgb];
            img_out[offset_bayer+1] = img[offset_rgb+3+1];
            img_out[offset_bayer+width] = img[offset_rgb+width*3+1];
            img_out[offset_bayer+width+1] = img[offset_rgb+width*3+3+2];
        }
        else if (bayer_pattern==4) {
            float lum = (img[offset_rgb+2] + img[offset_rgb+1] + img[offset_rgb])/3;
            img_out[offset_bayer] = int(lum*16);
            offset_rgb += 3;
            lum = (img[offset_rgb+2] + img[offset_rgb+1] + img[offset_rgb])/3;
            img_out[offset_bayer+1] = int(lum*16);
            offset_rgb += (width*3 - 3);
            lum = (img[offset_rgb+2] + img[offset_rgb+1] + img[offset_rgb])/3;
            img_out[offset_bayer+width] = img[offset_rgb+2]*16;
            offset_rgb += 3;
            lum = (img[offset_rgb+2] + img[offset_rgb+1] + img[offset_rgb])/3;
            img_out[offset_bayer+width+1] = int(lum*16);
        }
    }    
}

extern "C" __global__
void convert_C_R(const unsigned char* img,int width, int height,unsigned char* img_out) {
    //convert RGB to Bayer
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    if ((row < height) && (col < width)) {
        int offset_bayer = (row)*width + col;
        int offset_rgb = (row)*width*3 + col*3;
        int luminance = (img[offset_rgb+2] + img[offset_rgb+1] + img[offset_rgb])/3;
        img_out[offset_bayer] = abs(luminance - img[offset_rgb+2]);

    }    
}
extern "C" __global__
void Analyze_Raw(const unsigned short* img,int width, int height,unsigned short* img_out) {
    //convert RGB to Bayer
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    if ((row < height-1) && (col < width-1)) {
        int offset = (row)*width + col;
        int lum = (img[offset]+img[offset+1]+img[offset+width+1])/3;
        int red = img[offset+width];
        img_out[offset] = img[offset];
        img_out[offset+1] = img[offset+1];
        img_out[offset+width] = lum;
        img_out[offset+width+1] = img[offset+width+1];
    }    
}
extern "C" __global__
void Recover_Raw(const unsigned short* img,int width, int height,int max,unsigned short* img_out) {
    //convert RGB to Bayer
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    if ((row < height-1) && (col < width-1)) {
        int offset_bayer = (row)*width + col;
        int offset_rgb = (row)*width*3 + col*3;
        int lum = (img[offset_bayer]+img[offset_bayer+1]+img[offset_bayer+width+1])/3;
        int red = img[offset_bayer+width];
        int CR = lum - red;
        img_out[offset_rgb] = img[offset_bayer];
        img_out[offset_rgb+1] = img[offset_bayer];
        img_out[offset_rgb+2] = BOUND(img[offset_bayer] - CR,0,max);

        img_out[offset_rgb+3] = img[offset_bayer+1];
        img_out[offset_rgb+3+1] = img[offset_bayer+1];
        img_out[offset_rgb+3+2] = BOUND(img[offset_bayer+1] - CR,0,max);
        
        img_out[offset_rgb+width*3] = BOUND(CR+red,0,max);
        img_out[offset_rgb+width*3+1] = BOUND(CR+red,0,max);
        img_out[offset_rgb+width*3+2] = red;
        
        img_out[offset_rgb+width*3+3] = img[offset_bayer+width+1];
        img_out[offset_rgb+width*3+3+1] = img[offset_bayer+width+1];
        img_out[offset_rgb+width*3+3+2] = BOUND(img[offset_bayer+width+1] - CR,0,max);
    }    
}