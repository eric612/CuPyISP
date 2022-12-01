#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )

extern "C" __global__
void AdamsInterpolation(const short* in, int x, int y, int width, int direction, short* pix_out, short max_cut = 30) {
    if(direction == 0)
        //return BOUND(((in[(y + 1) * width + x] + in[(y - 1) * width + x])*0.5 + (in[y * width + x] * 2 - in[(y + 2) * width + x] - in[(y - 2) * width + x])*0.25),1,255);
        pix_out[0] = ((in[(y + 1) * width + x] + in[(y - 1) * width + x])*0.5 + BOUND((in[y * width + x] * 2 - in[(y + 2) * width + x] - in[(y - 2) * width + x]),-max_cut, max_cut)/4);
    else 
        //return BOUND(((in[y * width + x + 1] + in[y * width + x - 1])*0.5 + (in[y * width + x] * 2 - in[y  * width + x + 2] - in[y  * width + x - 2])*0.25), 1, 255);
        pix_out[0] = ((in[y * width + x + 1] + in[y * width + x - 1])*0.5 + BOUND((in[y * width + x] * 2 - in[y  * width + x + 2] - in[y  * width + x - 2]), -max_cut, max_cut)/4);

}
extern "C" __global__
void cfa_kernel(const short* source_image,int is_color, int row, int col,int width,short *pix_out) {
    float r,g,b;
    int r2,g2,b2;
    short img[5][5]; //5x5 crop_image
    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) {
            int offset = (row+i-2)*width + col + j -2;
            img[i][j] = source_image[offset];
        }
    }
    short out[4];
    if (is_color == 0) {
        r2 = img[2][2];
   
        g = 4 * img[2][2] - img[0][2] - img[2][0] - img[4][2] - img[2][4]
            + 2 * (img[3][2] + img[2][3] + img[1][2] + img[2][1]);
        b = 6 * img[2][2] - 3 * (img[0][2] + img[2][0] + img[4][2] + img[2][4]) / 2
            + 2 * (img[1][1] + img[1][3] + img[3][1] + img[3][3]);
        g2 = g / 8;
        b2 = b / 8;
    }
    
    else if (is_color == 1) {
        r = 5 * img[2][2] - img[2][0] - img[1][1] - img[3][1] - img[1][3] - img[3][3] - img[2][4]
            + ((img[0][2] + img[4][2]) / 2) + (4 * (img[2][1] + img[2][3]));
        g2 = img[2][2];
        b = 5 * img[2][2] - img[0][2] - img[1][1] - img[1][3] - img[4][2] - img[3][1] - img[3][3]
            + ((img[2][0] + img[2][4]) / 2) + (4 * (img[1][2] + img[3][2]));
        r2 = r / 8;
        b2 = b / 8;
    }
    else if (is_color == 2) {
        b = 5 * img[2][2] - img[2][0] - img[1][1] - img[3][1] - img[1][3] - img[3][3] - img[2][4]
            + (img[0][2] + img[4][2]) / 2 + 4 * (img[2][1] + img[2][3]);
        g2 = img[2][2];
        r = 5 * img[2][2] - img[0][2] - img[1][1] - img[1][3] - img[4][2] - img[3][1] - img[3][3]
            + (img[2][0] + img[2][4]) / 2 + 4 * (img[1][2] + img[3][2]);
        r2 = r / 8;
        b2 = b / 8;
    }
    else if (is_color == 3) {
        g = 4 * img[2][2] - img[0][2] - img[2][0] - img[4][2] - img[2][4]
            + 2 * (img[3][2] + img[2][3] + img[1][2] + img[2][1]);
        r = 6 * img[2][2] - 3 * (img[0][2] + img[2][0] + img[4][2] + img[2][4]) / 2
            + 2 * (img[1][1] + img[1][3] + img[3][1] + img[3][3]);
        b2 = img[2][2];
        r2 = r / 8;
        g2 = g / 8;
    }
    else if (is_color == 4) {
        g = img[2][3] + img[2][1] + img[1][2] + img[3][2] + img[1][1] + img[1][3] + img[3][1] + img[3][3];
        b = img[2][3] + img[2][1] + img[1][2] + img[3][2] + img[1][1] + img[1][3] + img[3][1] + img[3][3];
        r2 = img[2][2];
        g2 = g / 8;
        b2 = b / 8;
    }
    else if (is_color == 5) {
        AdamsInterpolation(source_image,col,row,width,1,out);
        r = out[0];
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r;
    }
    else if (is_color == 6) {
        AdamsInterpolation(source_image,col,row,width,0,out);
        r = out[0];
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r;
    }
    else if (is_color == 7) {
        AdamsInterpolation(source_image,col-1,row,width,0,&out[0]);
        AdamsInterpolation(source_image,col,row-1,width,1,&out[1]);
        AdamsInterpolation(source_image,col+1,row,width,0,&out[2]);
        AdamsInterpolation(source_image,col,row+1,width,1,&out[3]);
        //r = out[0] + out[1] + out[2] + out[3] + img[1][1] + img[1][3] + img[3][1] + img[3][3];
        r = out[0] + out[1] + out[2] + out[3];
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r/4;
    }    
    r2 = max(r2,0);
    r2 = min(r2,255*4);
    g2 = max(g2,0);
    g2 = min(g2,255*4);
    b2 = max(b2,0);
    b2 = min(b2,255*4);
    pix_out[0] = r2;
    pix_out[1] = g2;
    pix_out[2] = b2;
    
}
extern "C" __global__
void cfa(const short* img,int width, int height,int pad_w,int pad_h,int bayer_pattern,short* img_out) {

    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int i_width = width + pad_w;
    int i_height = height + pad_h;
    int pad_w2 = pad_w/2;
    int pad_h2 = pad_h/2;
    if ((row < i_height) && (col < i_width) && row>=0 && col>=0) {

        int offset = (row+2)*i_width + col + 2;
        int shift_offset = (row)*width*3 + col*3 ;
        short pix_out[3];
        if (bayer_pattern==0 ) {
                      
            cfa_kernel(img,0,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,1,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            cfa_kernel(img,2,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,3,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];             
            
        }
        else if (bayer_pattern==1 ) {           

            cfa_kernel(img,3,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,2,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            cfa_kernel(img,1,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,0,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];                
        }
        else if (bayer_pattern==2 ) {          

            cfa_kernel(img,2,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,3,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            cfa_kernel(img,0,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,1,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];              
        }
        else if (bayer_pattern==3 ) {          

            cfa_kernel(img,1,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,0,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            cfa_kernel(img,3,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,2,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];             
        }        
        else if (bayer_pattern==4 ) {    
            cfa_kernel(img,6,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,7,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            cfa_kernel(img,4,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            cfa_kernel(img,5,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];          
            /*
            //cfa_kernel(img,0,row+2,col+2,i_width,pix_out);   
            //int pix = (img[offset]+img[offset+1]+img[offset+i_width]+img[offset+i_width+1])/4;
            img_out[shift_offset] = img[offset];
            img_out[shift_offset+1] = img[offset];
            img_out[shift_offset+2] = img[offset];
            
            //cfa_kernel(img,1,row+2,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = img[offset+1];
            img_out[shift_offset+1] = img[offset+1];
            img_out[shift_offset+2] = img[offset+1];  
            
            //cfa_kernel(img,2,row+3,col+2,i_width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = img[offset+i_width];
            img_out[shift_offset+1] = img[offset+i_width];
            img_out[shift_offset+2] = img[offset+i_width];
            
            //cfa_kernel(img,0,row+3,col+3,i_width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = img[offset+i_width+1];
            img_out[shift_offset+1] = img[offset+i_width+1];
            img_out[shift_offset+2] = img[offset+i_width+1];
            */
        } 
    }
}
