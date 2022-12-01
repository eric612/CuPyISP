extern "C" __global__
void gac(const short* img,int width, int height,const short* lut,int divide,short* img_out) {
  
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = row*width*3 + col*3 ;
        
        img_out[offset] =  lut[img[offset]]/divide;
        img_out[offset+1] =  lut[img[offset+1]]/divide;
        img_out[offset+2] =  lut[img[offset+2]]/divide;
        
    }    
}
extern "C" __global__
void slut(unsigned short* lut,double gamma,int maxval) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //lut[index] = round(pow(float(index)/float(maxval), gamma)* maxval);
    //lut[index+1] = round(pow(float(index+1)/float(maxval), gamma)* maxval);
    //lut[index+2] = round(pow(float(index+2)/float(maxval), gamma)* maxval);
    //lut[index+3] = round(pow(float(index+3)/float(maxval), gamma)* maxval);
    lut[index] = round(pow(float(index)/float(maxval), float(gamma))* maxval);

}