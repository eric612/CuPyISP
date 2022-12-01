extern "C" __global__
void cnd(const short* img, int row, int col, int width,int thres,int &is_noise,short &avgG,short &avgC1,short &avgC2){

    avgG = 0;
    avgC1 = 0;
    avgC2 = 0;
    is_noise = 0;
    for(int i=-4;i<4;i++) {
        for(int j=-4;j<4;j++) {
            int offset = (row+i)*width + col + j;
            if (i % 2 == 1 && j % 2 == 0) {
                avgG = avgG + img[offset];
            }
            else if (i % 2 == 0 && j % 2 == 1) {
                avgG = avgG + img[offset];
            }
            else if (i % 2 == 0 && j % 2 == 0) {
                avgC1 = avgC1 + img[offset];
            }
            else if (i % 2 == 1 && j % 2 == 1) {
                avgC2 = avgC2 + img[offset];
            }
        }
    }
    avgG = avgG / 40;
    avgC1 = avgC1 / 25;
    avgC2 = avgC2 / 16;
    int offset = (row+4)*width + col+4 ;
    short center = img[offset];
    if (center > (avgG) + thres && center > (avgC2) + thres) {
        if ((avgC1) > (avgG) + thres && (avgC1) > (avgC2) + thres) {
            is_noise = 1;
        }
        else {
            is_noise = 0;
        }
    }
    else {
        is_noise = 0;    
    }
}
extern "C" __global__
void cnc(int is_color, short center, short avgG, short avgC1, short avgC2,float r_gain,float b_gain,float &center_out) {
    float fade1,fade2;
    float dampFactor = 1.0;
    int signalGap = center - max(avgG, avgC2);
    float signalMeter;
    if (is_color == 0) {
        if (r_gain <= 1.0)
            dampFactor = 1.0;
        else if (r_gain > 1.0 && r_gain <= 1.2)
            dampFactor = 0.5;
        else if (r_gain > 1.2)
            dampFactor = 0.3;
    }
    else if (is_color == 2) {
        if (b_gain <= 1.0)
            dampFactor = 1.0;
        else if (b_gain > 1.0 && b_gain <= 1.2)
            dampFactor = 0.5;
        else if (b_gain > 1.2)
            dampFactor = 0.3;
    }
    float chromaCorrected = max(avgG, avgC2) + dampFactor * signalGap;
    if (is_color == 0)
        signalMeter = 0.299 * avgC1 + 0.587 * avgG + 0.114 * avgC2;
    else if (is_color == 2)
        signalMeter = 0.299 * avgC2 + 0.587 * avgG + 0.114 * avgC1;
    if (signalMeter <= 30)
        fade1 = 1.0;
    else if (signalMeter > 30 && signalMeter <= 50)
        fade1 = 0.9;
    else if (signalMeter > 50 && signalMeter <= 70)
        fade1 = 0.8;
    else if (signalMeter > 70 && signalMeter <= 100)
        fade1 = 0.7;
    else if (signalMeter > 100 && signalMeter <= 150)
        fade1 = 0.6;
    else if (signalMeter > 150 && signalMeter <= 200)
        fade1 = 0.3;
    else if (signalMeter > 200 && signalMeter <= 250)
        fade1 = 0.1;
    else
        fade1 = 0;
    if (avgC1 <= 30)
        fade2 = 1.0;
    else if (avgC1 > 30 && avgC1 <= 50)
        fade2 = 0.9;
    else if (avgC1 > 50 && avgC1 <= 70)
        fade2 = 0.8;
    else if (avgC1 > 70 && avgC1 <= 100)
        fade2 = 0.6;
    else if (avgC1 > 100 && avgC1 <= 150)
        fade2 = 0.5;
    else if (avgC1 > 150 && avgC1 <= 200)
        fade2 = 0.3;
    else if (avgC1 > 200)
        fade2 = 0;
    float fadeTot = fade1 * fade2;
    
    center_out = (1.0 - fadeTot) * center + fadeTot * chromaCorrected;
}
extern "C" __global__
void cnf_kernel(const short* img,int is_color, int row, int col,int width,int thres,float r_gain,float b_gain,short &pix_out) {
    int is_noise;
    short avgG;
    short avgC1;
    short avgC2; 
    cnd(img,row,col,width,thres,is_noise,avgG,avgC1,avgC2);
    int offset = row*width + col;
    pix_out = img[offset];
    short center = img[offset];
    float center_out;
    if (is_noise)
        cnc(is_color, center, avgG, avgC1, avgC2,r_gain,b_gain,center_out);
}
extern "C" __global__
void cnf(const short* img,int width, int height,int pad_w,int pad_h,int filter_w,int filter_h,int thres,float r_gain,float b_gain,int bayer_pattern,short* img_out) {

    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int o_width = width - pad_w;
    if ((row < height) && (col < width)) {

        int offset = (row+4)*width + col + 4;
        int shift_offset = (row)*o_width + col ;
        //img_out[shift_offset] = img[offset];
        //img_out[shift_offset+1] = img[offset+1];
        //img_out[shift_offset+width] = img[offset+width];
        //img_out[shift_offset+width+1] = img[offset+width+1];
        if (bayer_pattern==0 ) {
            short r = img[offset];
            short gr = img[offset+1];
            short gb = img[offset+width];
            short b = img[offset+width+1];            
            cnf_kernel(img,0,row+4,col+4,width,thres,r_gain,b_gain,r);
            cnf_kernel(img,2,row+5,col+5,width,thres,r_gain,b_gain,b);
            img_out[shift_offset] = r;
            img_out[shift_offset+1] = gr;
            img_out[shift_offset+o_width] = gb;
            img_out[shift_offset+o_width+1] = b;                  
        }
        else if (bayer_pattern==1 ) {
            short b = img[offset];
            short gb = img[offset+1];
            short gr = img[offset+width];
            short r = img[offset+width+1];            
            cnf_kernel(img,0,row+4,col+4,width,thres,r_gain,b_gain,b);
            cnf_kernel(img,2,row+5,col+5,width,thres,r_gain,b_gain,r);
            img_out[shift_offset] = b;
            img_out[shift_offset+1] = gb;
            img_out[shift_offset+o_width] = gr;
            img_out[shift_offset+o_width+1] = r;                 
        }
        else if (bayer_pattern==2 ) {
            short gb = img[offset];
            short b = img[offset+1];
            short r = img[offset+width];
            short gr = img[offset+width+1];            
            cnf_kernel(img,0,row+4,col+5,width,thres,r_gain,b_gain,b);
            cnf_kernel(img,2,row+5,col+4,width,thres,r_gain,b_gain,r);
            img_out[shift_offset] = gb;
            img_out[shift_offset+1] = b;
            img_out[shift_offset+o_width] = r;
            img_out[shift_offset+o_width+1] = gr;             
        }
        else if (bayer_pattern==3 ) {
            short gr = img[offset];
            short r = img[offset+1];
            short b = img[offset+width];
            short gb = img[offset+width+1];            
            cnf_kernel(img,0,row+4,col+5,width,thres,r_gain,b_gain,r);
            cnf_kernel(img,2,row+5,col+4,width,thres,r_gain,b_gain,b);
            img_out[shift_offset] = gr;
            img_out[shift_offset+1] = r;
            img_out[shift_offset+o_width] = b;
            img_out[shift_offset+o_width+1] = gb;              
        }        

    }
}
