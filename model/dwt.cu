extern "C" __global__
void dwt_h(const short* img,int width, int height,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width + col;
        int offset2 = row*width + col*2;
        int offset3 = row*width + col + width/2;
        img_out[offset] = (img[offset2]+img[offset2+1]);
        img_out[offset3] = (img[offset2]-img[offset2+1]);
        
    }    
}

extern "C" __global__
void dwt_v(const short* img,int width, int height,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width + col;
        int offset2 = row*width*2 + col;
        int offset3 = row*width + col + width*(height/2);
        img_out[offset] = (img[offset2]+img[offset2+width]);
        img_out[offset3] = (img[offset2]-img[offset2+width]);
        
    }    
}

extern "C" __global__
void idwt_h(const short* img,int width, int height,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width + col;
        int offset2 = row*width + col*2;
        int offset3 = row*width + col + width/2;
        img_out[offset2] = (img[offset] + img[offset3])/2;
        img_out[offset2+1] = img[offset] - img_out[offset2];
        //img_out[offset] = (img[offset2]+img[offset2+1]);
        //img_out[offset3] = (img[offset2]-img[offset2+1]);
        
    }    
}

extern "C" __global__
void idwt_v(const short* img,int width, int height,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width + col;
        int offset2 = row*width*2 + col;
        int offset3 = row*width + col + width*(height/2);
        img_out[offset2] = (img[offset] + img[offset3])/2;
        img_out[offset2+width] = img[offset] - img_out[offset2];
        //img_out[offset] = (img[offset2]+img[offset2+width]);
        //img_out[offset3] = (img[offset2]-img[offset2+width]);
        
    }    
}