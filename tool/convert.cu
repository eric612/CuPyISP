

extern "C" __global__
void convert(const unsigned char* img,int width, int height,int bayer_pattern,unsigned char* img_out) {
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
    }    
}
