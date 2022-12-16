#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )

extern "C" __global__
void AdamsInterpolation(const short* in, int x, int y, int width, int direction, short* pix_out, short max_cut = 30) {
    if(direction == 0)
        //return BOUND(((in[(y + 1) * width + x] + in[(y - 1) * width + x])*0.5 + (in[y * width + x] * 2 - in[(y + 2) * width + x] - in[(y - 2) * width + x])*0.25),1,255);
        pix_out[0] = (in[(y + 1) * width + x] + in[(y - 1) * width + x])*0.5;// + BOUND((in[y * width + x] * 2 - in[(y + 2) * width + x] - in[(y - 2) * width + x])*0.25,-30,30);
    else 
        //return BOUND(((in[y * width + x + 1] + in[y * width + x - 1])*0.5 + (in[y * width + x] * 2 - in[y  * width + x + 2] - in[y  * width + x - 2])*0.25), 1, 255);
        pix_out[0] = (in[y * width + x + 1] + in[y * width + x - 1])*0.5;// + BOUND((in[y * width + x] * 2 - in[y  * width + x + 2] - in[y  * width + x - 2]*0.25), -30, 30);

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
        int h = img[2][2]*2 - img[2][0] - img[2][4];
        int v = img[2][2]*2 - img[0][2] - img[4][2];
        if (abs(h)>abs(v)) {
            g2 = (g+v) / 8;
            b2 = (b+v) / 8;
        }
        else {
            g2 = (g+h) / 8;
            b2 = (b+h) / 8;            
        }
    }
    else if (is_color == 5) {
        AdamsInterpolation(source_image,col,row,width,1,out);
        int h = (img[2][2]*2 - img[2][0] - img[2][4])/4;
        int v = (img[2][2]*2 - img[0][2] - img[4][2])/4;
        if (abs(h)>abs(v)) {
            r = (out[0]+v);
        }
        else {
            r = (out[0]+h);          
        }        
        //r = out[0];
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r;
    }
    else if (is_color == 6) {
        AdamsInterpolation(source_image,col,row,width,0,out);
        int h = (img[2][2]*2 - img[2][0] - img[2][4])/4;
        int v = (img[2][2]*2 - img[0][2] - img[4][2])/4;
        if (abs(h)>abs(v)) {
            r = (out[0]+v);
        }
        else {
            r = (out[0]+h);          
        }
        //r = out[0];
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r;
    }
    else if (is_color == 7) {
        AdamsInterpolation(source_image,col-1,row,width,0,&out[0]);
        AdamsInterpolation(source_image,col,row-1,width,1,&out[1]);
        AdamsInterpolation(source_image,col+1,row,width,0,&out[2]);
        AdamsInterpolation(source_image,col,row+1,width,1,&out[3]);
        int h = (img[2][2]*2 - img[2][1] - img[2][3]);
        int v = (img[2][2]*2 - img[1][2] - img[3][2]);
        if (abs(h)>abs(v)) {
            r = (out[0] + out[1] + out[2] + out[3] );
        }
        else {
            r = (out[0] + out[1] + out[2] + out[3] );          
        }
        //r = out[0] + out[1] + out[2] + out[3] + img[1][1] + img[1][3] + img[3][1] + img[3][3];
        
        g2 = img[2][2];
        b2 = img[2][2];
        r2 = r/4;
    }    
    r2 = max(r2,0);
    r2 = min(r2,4095);
    g2 = max(g2,0);
    g2 = min(g2,4095);
    b2 = max(b2,0);
    b2 = min(b2,4095);
    pix_out[0] = r2;
    pix_out[1] = g2;
    pix_out[2] = b2;
    
}
extern "C" __global__
void Calc_CR(const short* img,int is_color, int row, int col,int width,short *pix_out) {
    int lum = 0;
    int red = 0;
    int offset = (row)*width + col ;
    if (is_color == 0) {
        int h = abs(img[offset+1]*2 - img[offset+1+width] - img[offset+1-width])+abs(img[offset-1]*2- img[offset-1-width]- img[offset-1+width])+abs((img[offset+width] - img[offset-width]));
        int v = abs(img[offset+width]*2 - img[offset+width-1] - img[offset+width+1])+abs(img[offset-width]*2 - img[offset-width-1] - img[offset-width+1]) + abs(img[offset+1] - img[offset-1]);
        int l1 = abs(img[offset+1] - img[offset-width]);
        int l2 = abs(img[offset+width] - img[offset-1]);
        int l3 = abs(img[offset+width] - img[offset+1]);
        int l4 = abs(img[offset-1] - img[offset-width]);
        int l5 = abs(img[offset+width+1] + img[offset-width-1]);
        int l6 = abs(img[offset+width-1] + img[offset-width+1]);
        
        int gradient[10],interp[10];
        gradient[0] = h;
        gradient[1] = v;
        gradient[2] = l1+l2+l5;
        gradient[3] = l3+l4+l6;
        
        
        interp[0] = (img[offset+width] + img[offset-width])/2;
        interp[1] = (img[offset+1] + img[offset-1])/2;
        interp[2] = (img[offset+width+1] + img[offset-width-1])/2;
        interp[3] = (img[offset+width-1] + img[offset-width+1])/2;
        

        int minima = 65536;
        int index = 0;
        for (int i =0;i<4;i++) {
            if(gradient[i]<minima) {
                minima = gradient[i];
                index = i;
            }
        }
        lum = interp[index];
        int avg = (interp[0]+interp[1])/2;
        int laplacian = abs(avg - img[offset+width]) + abs(avg - img[offset-width]) + abs(avg - img[offset+1]) + abs(avg - img[offset-1]);
        if(laplacian<=minima) {
            //lum = avg;
        }
        /*
        if (v<h && v<l1 && v<l2) {
            lum = (img[offset+1] + img[offset-1])/2;
        }
        else if (h<v && h<l1 && h<l2) {            
            lum = (img[offset+width] + img[offset-width])/2;
        }
        else if (l1<v && l1<h && l1<l2) {            
            lum = (img[offset+width+1] + img[offset-width-1])/2;
        }
        else {            
            lum = (img[offset-width+1] + img[offset+width-1])/2;
        }*/
        //lum = 0;
        //lum = lum = (img[offset+width] + img[offset-width])/2;
        //lum = (img[offset+width] + img[offset-width])/2;
        red = img[offset];
    }
    else if (is_color == 1) {
        lum = img[offset];
        //lum = (img[offset]*8 + img[offset-width]*2 + img[offset+width]*2 + img[offset-width-1] + img[offset-width+1] + img[offset+width-1] + img[offset+width+1])/16;
        red = (img[offset-1] + img[offset+1])/2;
    }
    else if (is_color == 2) {
        lum = img[offset];
        //lum = (img[offset]*8 + img[offset-1]*2 + img[offset+1]*2 + img[offset-width-1] + img[offset-width+1] + img[offset+width-1] + img[offset+width+1])/16;
        red = (img[offset-width] + img[offset+width])/2;
    }
    else if (is_color == 3) {
        lum = img[offset];
        //lum = (img[offset]*4 + img[offset-width] + img[offset+width] + img[offset-1] + img[offset+1])/8;
        red = (img[offset-width-1] + img[offset-width+1] + img[offset+width-1] + img[offset+width+1])/4;
        /*
        int h = abs(img[offset]*2 - img[offset-width] - img[offset+width]);
        int v = abs(img[offset]*2 - img[offset-1] - img[offset+1]);
        if (h>v) {
            //red = (img[offset+width-1] + img[offset+width+1])/2;
        }
        else {
            
            //red = (img[offset-width+1] + img[offset+width+1])/2;
        }*/        
        
    }
    pix_out[0] = lum - red;
}
extern "C" __global__
void RCCC_kernel(const short* source_image,const short* pre_maps,int is_color, int row, int col,int width,int map_width,short *pix_out) {
    float r,g,b;
    int r2,g2,b2;
    short img[5][5]; //5x5 crop_image
    short CR_img[5][5]; //5x5 crop_image
    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) {
            int offset = (row+i-2)*width + col + j -2;
            int offset2 = (row+i-2)*map_width + col + j -2;
            img[i][j] = source_image[offset];
            CR_img[i][j] = pre_maps[offset2];
        }
    }
    short out[4];
    int lum = 0;
    int CR = 0; // lum - R
    int red = 0;
    int offset_map = (row-2)*map_width + col-2 ;
    int th = 512;
    if (is_color == 0) {         
        red = img[2][2];
        CR = pre_maps[offset_map];
        r2 = red;
        g2 = BOUND(CR+red,0,4095);
        b2 = BOUND(CR+red,0,4095);
    }
    else if (is_color == 1) {
        lum = img[2][2];
        int minimum = 9999;
        int red = (img[2][1]*2 + img[2][3]*2 + img[0][1] + img[4][1] + img[0][3] + img[4][3])/8;
        int tmp = lum - red;     
        int count = 0;
        int sum = 0;
        for(int i=0;i<5;i++) {
            for(int j=0;j<5;j++) {
                if( (i%2==0) && (j%2==1)) {
                    count++;
                    sum += CR_img[i][j];
                }
                else if (abs(CR_img[i][j] - CR_img[2][2])<th) {
                    count++;
                    sum += CR_img[i][j];                    
                }
            }
        }
        if(count!=0) {
            tmp = sum/count;
        }
        //red = (img[2][1]*2 + img[2][3]*2 + img[0][1] + img[4][1] + img[0][3] + img[4][3])/8;                  
        r2 = BOUND(lum - tmp,0,4095);
        g2 = BOUND(lum,0,4095);
        b2 = BOUND(lum,0,4095);        
    }
    else if (is_color == 2) {
        lum = img[2][2];
        int minimum = 9999;
        int red = (img[1][2]*2 + img[3][2]*2 + img[1][0] + img[1][4] + img[3][0] + img[3][4])/8; 
        int tmp = lum - red;   
        int count = 0;
        int sum = 0;        
        for(int i=0;i<5;i++) {
            for(int j=0;j<5;j++) {
                if( (i%2==1) && (j%2==0)) {
                    count++;
                    sum += CR_img[i][j];
                }
                else if (abs(CR_img[i][j] - CR_img[2][2])<th) {
                    count++;
                    sum += CR_img[i][j];                    
                }                
            }
        }
        if(count!=0) {
            tmp = sum/count;
        }
        //red = (img[1][2]*2 + img[3][2]*2 + img[1][0] + img[1][4] + img[3][0] + img[3][4])/8;
        r2 = BOUND(lum - tmp,0,4095);
        g2 = BOUND(lum,0,4095);
        b2 = BOUND(lum,0,4095);  
    }
    else if (is_color == 3) {
        lum = img[2][2];
        int minimum = 9999;
        int red = (img[1][1] + img[1][3] + img[3][1] + img[3][3])/4;
        int tmp = lum - red;  
        int count = 0;
        int sum = 0;        
        for(int i=0;i<5;i++) {
            for(int j=0;j<5;j++) {
                if( (i%2==1) && (j%2==1)) {
                    count++;
                    sum += CR_img[i][j];
                }
                else if (abs(CR_img[i][j] - CR_img[2][2])<th) {
                    count++;
                    sum += CR_img[i][j];                    
                }                
            }
        }
        if(count!=0) {
            tmp = sum/count;
        }        
        //red = (img[1][1] + img[1][3] + img[3][1] + img[3][3])/4;        
        r2 = BOUND(lum - tmp,0,4095);
        g2 = BOUND(lum,0,4095);
        b2 = BOUND(lum,0,4095);
    }
    /*
    r2 = max(r2,0);
    r2 = min(r2,4095);
    g2 = max(g2,0);
    g2 = min(g2,4095);
    b2 = max(b2,0);
    b2 = min(b2,4095);*/
    //b2 = (0.413*g2-0.299*r2)/0.114;
    //g2 = (g2 - 0.299*r2 - 0.114*b2)/0.587;
    pix_out[0] = BOUND(b2,0,4095);
    pix_out[1] = BOUND(g2,0,4095);
    pix_out[2] = BOUND(r2,0,4095);
}
extern "C" __global__
void pre_maps(const short* img,int width, int height,int pad_w,int pad_h,int bayer_pattern,short* img_out) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int i_width = width + pad_w;
    int i_height = height + pad_h;
    int pad_w2 = pad_w/2;
    int pad_h2 = pad_h/2;
    if ((row < i_height) && (col < i_width) && row>=0 && col>=0) {

        int offset = (row+2)*i_width + col + 2;
        int shift_offset = (row)*width + col ;
        short pix_out[3];
        if (bayer_pattern==4 ) {
            Calc_CR(img,2,row+2,col+2,i_width,pix_out);            
            img_out[shift_offset] = (pix_out[0]);
            
            Calc_CR(img,3,row+2,col+3,i_width,pix_out);
            shift_offset +=1 ;
            img_out[shift_offset] = (pix_out[0]);
            
            Calc_CR(img,0,row+3,col+2,i_width,pix_out);
            shift_offset += (width-1);
            img_out[shift_offset] = (pix_out[0]);
            
            Calc_CR(img,1,row+3,col+3,i_width,pix_out);
            shift_offset +=1 ;
            img_out[shift_offset] = (pix_out[0]);              
        }
    }        
}
extern "C" __global__
void cfa(const short* img,const short* pre_maps,int width, int height,int pad_w,int pad_h,int bayer_pattern,short* img_out) {

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
            RCCC_kernel(img,pre_maps,2,row+2,col+2,i_width,width,pix_out);            
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            RCCC_kernel(img,pre_maps,3,row+2,col+3,i_width,width,pix_out);
            shift_offset +=3 ;
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];  
            
            RCCC_kernel(img,pre_maps,0,row+3,col+2,i_width,width,pix_out);
            shift_offset += (width*3-3);
            img_out[shift_offset] = pix_out[0];
            img_out[shift_offset+1] = pix_out[1];
            img_out[shift_offset+2] = pix_out[2];
            
            RCCC_kernel(img,pre_maps,1,row+3,col+3,i_width,width,pix_out);
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
