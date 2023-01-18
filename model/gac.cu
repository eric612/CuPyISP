#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )
extern "C" __global__
void gac(const short* img,int width, int height,const short* lut,int divide,int maxval,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width*3 + col*3 ;
        int pix = BOUND(img[offset],0,maxval);
        img_out[offset] =  lut[pix]/divide;
        pix = BOUND(img[offset+1],0,maxval);
        img_out[offset+1] =  lut[pix]/divide;
        pix = BOUND(img[offset+2],0,maxval);
        img_out[offset+2] =  lut[pix]/divide;
        
    }    
}

extern "C" __global__
void slut(unsigned short* lut,double gamma,int maxval) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //lut[index] = round(pow(float(index)/float(maxval), gamma)* maxval);
    //lut[index+1] = round(pow(float(index+1)/float(maxval), gamma)* maxval);
    //lut[index+2] = round(pow(float(index+2)/float(maxval), gamma)* maxval);
    //lut[index+3] = round(pow(float(index+3)/float(maxval), gamma)* maxval);
    lut[index] = BOUND(round(pow(float(index)/float(maxval), float(gamma))* maxval),0,maxval);

}