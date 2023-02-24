extern "C" __global__
void average(const short* array,int size,int &out) {
    int sum = 0;
    for(int i=0;i<size;i++) {
        sum = sum + array[i];
    }
    out = sum/size;
}
extern "C" __global__
void average2(const int* array,int size,int &out) {
    int sum = 0;
    for(int i=0;i<size;i++) {
        sum = sum + array[i];
    }
    out = sum/size;
}
extern "C" __global__
void subtract_power(const short* array1,const short* array2,int size,int *out) {

    for(int i=0;i<size;i++) {
        out[i] = (array1[i] - array2[i])*(array1[i] - array2[i]);
    }
}
extern "C" __global__
void weighted_sum(const short* array,const short* weight,int size,int &sum) {
    sum = 0;
    for(int i=0;i<size;i++) {
        sum = sum + weight[i]*array[i];
    }
}
extern "C" __global__
void patch(const short* source_image,int filter_size, int row, int col,int width,short *patch) {
    int fw = filter_size;
    int shift = fw/2;
    for(int i=0;i<fw;i++) {
        for(int j=0;j<fw;j++) {
            int offset = (row+i-shift)*width + col + j - shift;
            patch[i*fw+j] = source_image[offset];
        }
    }       
}
extern "C" __global__
void nlm_kernel(const short* source_image, int row, int col,int distance,int filter_size,int width,int h,short &pix_out) {
    int offset = distance/2;
    short center[9]; //3x3 crop_image
    short out[9]; //3x3 crop_image
    int sub[9];
    float average = 0;
    float weight = 0;
    float max_weight = 0;
    float wmax;
    patch(source_image,filter_size,row,col,width,center);
    
    for(int i=0;i<distance;i++) {
        for(int j=0;j<distance;j++) {
            int avg;
            int y = row + i - offset;
            int x = col + j - offset;
            //int offset = (y)*width + x;           
            patch(source_image,filter_size,y,x,width,out);
            subtract_power(out,center,9,sub);
            average2(sub,9,avg);
            float w = exp(-avg/100.0);   
            if (w > wmax)
                wmax = w;
            weight = weight + w;
            average = average + w * source_image[y*width+x];            
        }
    }
    average = average + wmax*source_image[row*width+col];
    weight = weight + wmax;
    pix_out = short(average / weight);
    
}

extern "C" __global__
void nlm(const short* img,int width, int height,int pad_w,int pad_h,int ds,int ks,int h,short* img_out) {
    // Non-local mean 
    // ks : kernel size
    // ds : search distance
    
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    int i_width = width + pad_w;
    int i_height = height + pad_h;
    int pad_w2 = pad_w/2;
    int pad_h2 = pad_h/2;
    int avg;
    short out[9]; //3x3 crop_image
    short pix_out;
    if ((row < height && row>=0) && (col < width && col>=0)) {
        int offset = (row+pad_w2)*i_width + col + pad_w2;
        int shift_offset = (row)*width + col ;
        nlm_kernel(img,row+pad_h2,col+pad_w2,ds,ks,i_width,h,pix_out);
        //patch(img,3,row+pad_h2,col+pad_w2,i_width,out);
        //average(out,9,avg);
        img_out[shift_offset] = pix_out;
        /*
        nlm_kernel(img,row+pad_h2,col+pad_w2+1,ds,ks,i_width,h,pix_out);
        img_out[shift_offset+1] = pix_out;
        nlm_kernel(img,row+pad_h2+1,col+pad_w2,ds,ks,i_width,h,pix_out);
        img_out[shift_offset+width] = pix_out;
        nlm_kernel(img,row+pad_h2+1,col+pad_w2+1,ds,ks,i_width,h,pix_out);
        img_out[shift_offset+width+1] = pix_out;*/
        /*patch(img,3,row+pad_h2,col+pad_w2,i_width,out);
        average(out,9,avg);
        img_out[shift_offset] = avg;
        //img_out[shift_offset] = img[offset];
        patch(img,3,row+pad_h2,col+pad_w2+1,i_width,out);
        average(out,9,avg);
        img_out[shift_offset+1] = avg;
        patch(img,3,row+pad_h2+1,col+pad_w2,i_width,out);
        average(out,9,avg);
        img_out[shift_offset+width] = avg;
        patch(img,3,row+pad_h2+1,col+pad_w2+1,i_width,out);
        average(out,9,avg);
        img_out[shift_offset+width+1] = avg;*/
    }    
}
