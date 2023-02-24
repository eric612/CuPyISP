extern "C" __global__
void filter_5x5(const short* source_image, int row, int col,const double* filter,int width,short &pix_out) {
    int offset = 5/2;
    float average = 0.;
    float weight = 0.;
    int shift_offset = (row)*width + col ;
    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) {
            int avg;
            int y = row + i - offset;
            int x = col + j - offset;
            average = average + filter[i*5+j]*source_image[y*width+x];  
            weight = weight + filter[i*5+j];
        }
    }
    //pix_out = source_image[shift_offset];
    pix_out = int(average/weight);
}

extern "C" __global__
void aaf(const short* img,int width, int height,int pad_w,int pad_h,const double* filter,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    int pad_w2 = pad_w/2;
    int pad_h2 = pad_h/2;
    int i_width = width + pad_w;
    int i_height = height + pad_h;
    short pix_out;
    //if ((row < height-pad_h2 && row>=pad_h2) && (col < width-pad_w2 && col>=pad_w2)) {
        int offset = row*width + col ;
        int shift_offset = (row+pad_h2)*i_width + col + pad_w2;
        //img_out[offset] = img[shift_offset];
        filter_5x5(img,row+pad_h2,col+pad_w2,filter,i_width,pix_out);
        img_out[offset] = pix_out;
    //}    
}
