#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b)) 
#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )
extern "C" __global__
void SortArray(int *array, int size)
{	
	int i, j, n=size, tmp;
	for (i=0; i<n-1; i++) {
		for (j=0; j<n-1-i; j++)
			if (array[j+1] < array[j]) {  
				tmp = array[j];         
				array[j] = array[j+1];
				array[j+1] = tmp;			
			}			
	}
}
extern "C" __global__
void DPC_kernel(const short* source_image,int is_color, int row, int col,int width,int thres,int nr_thres,short *pix_out) {
    short AVGPIX[4];
    int minavg;
    int maxavg;
    short p[25]; //5x5 crop_image
    int mx,mn,lu,ru,ld,rd;
    int q[13];
    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) {
            int index = (row+i-2)*width + col + j - 2;
            p[i*5+j] = source_image[index];
        }
    }
    if (is_color == 0) {         
        //DPC
        AVGPIX[0]=(p[10]+p[14])>>1;//RL
        AVGPIX[1]=(p[2]+p[22])>>1;//UD
        AVGPIX[2]=(p[0]+p[24])>>1;//D1
        AVGPIX[3]=(p[4]+p[20])>>1;//D2
        
        //DIFPIX[0]=abs(p[10]-p[14]);//RL
        //DIFPIX[1]=abs(p[2]-p[22]);//UD
        //DIFPIX[2]=abs(p[0]-p[24]);//D1
        //DIFPIX[3]=abs(p[4]-p[20]);//D2
        
        minavg=MIN(MIN(AVGPIX[0], AVGPIX[1]), MIN(AVGPIX[2], AVGPIX[3])); 
        maxavg=MAX(MAX(AVGPIX[0], AVGPIX[1]), MAX(AVGPIX[2], AVGPIX[3])); 
                                
        q[0]=p[0];
        q[1]=p[2];
        q[2]=p[4];
        q[3]=p[10];
        q[4]=p[12];
        q[5]=p[14];
        q[6]=p[20];
        q[7]=p[22];
        q[8]=p[24];
        
        SortArray(q, 9);
        pix_out[0] = p[12];
        //if( ((p[12]-maxavg>=thres) && (q[8]-p[12]<=nr_thres && p[12]-q[7]>=thres)) )//2011/0607 start
        //{
        //    pix_out[0] = maxavg;
        //}
        //else if( (minavg-p[12]>=DpcThd2) && (q[0]==p[12] && q[1]-p[12]>=DpcThd2) )
        if( ((minavg-p[12]>=thres) && (p[12]-q[0]<=nr_thres && q[1]-p[12]>=thres)) )//2011/0607 start
        {
            pix_out[0] = minavg;
        }
    }
    else if (is_color == 1) {
        //DPC
        int INMAX, INMIN, OUTMAX, OUTMIN;
        INMAX=MAX(MAX(p[6], p[8]), MAX(p[16], p[18]));
        INMIN=MIN(MIN(p[6], p[8]), MIN(p[16], p[18]));
        OUTMAX=MAX(MAX(p[0], p[4]), MAX(p[20], p[24]));
        OUTMIN=MIN(MIN(p[0], p[4]), MIN(p[20], p[24]));
        if(abs(p[12]-INMAX)<=nr_thres || abs(INMIN-p[12])<=nr_thres)
        {
            AVGPIX[0]=(p[10]+p[14])>>1;//RL
            AVGPIX[1]=(p[2]+p[22])>>1;//UD
            AVGPIX[2]=(p[0]+p[24])>>1;//D1
            AVGPIX[3]=(p[4]+p[20])>>1;//D2					
        }
        else
        {
            AVGPIX[0]=(p[10]+p[14])>>1;//RL
            AVGPIX[1]=(p[2]+p[22])>>1;//UD
            AVGPIX[2]=(p[6]+p[18])>>1;//D1
            AVGPIX[3]=(p[8]+p[16])>>1;//D2												
        }
        minavg=MIN(MIN(AVGPIX[0], AVGPIX[1]), MIN(AVGPIX[2], AVGPIX[3])); 
        maxavg=MAX(MAX(AVGPIX[0], AVGPIX[1]), MAX(AVGPIX[2], AVGPIX[3])); 
        
        if(1)
        {
                
            //0614 modify						
            q[0]=p[2];
            q[1]=p[6];
            q[2]=p[8];
            q[3]=p[10];
            q[4]=p[12];
            q[5]=p[14];
            q[6]=p[16];
            q[7]=p[18];
            q[8]=p[22];
                
            SortArray(q, 9);
                
            lu=(2*p[6]-p[0]);
            ru=(2*p[16]-p[20]);
            ld=(2*p[18]-p[24]);
            rd=(2*p[8]-p[4]);

            mx=MAX(MAX(lu, ru), MAX(ld, rd));
            mn=MIN(MIN(lu, ru), MIN(ld, rd));
            pix_out[0] = p[12];                           
                //if( (p[12]-maxavg>=DpcThd2) && (q[8]-p[12]<=DpcNRThd && p[12]-q[6]>=DpcThd2) )
            //if( ((p[12]-maxavg>=thres && p[12]>mx) || ((p[12]-maxavg>=thres) && (q[8]-p[12]<=nr_thres && p[12]-q[6]>=thres))) )//2011/0607 start
            //{
            //    pix_out[0] = maxavg;
            //}					
            //else if( (minavg-p[12]>=DpcThd2) && (p[12]-q[0]<=DpcNRThd && q[2]-p[12]>=DpcThd2) )
            if( ((minavg-p[12]>=thres && p[12]<mn) || ((minavg-p[12]>=thres) && (p[12]-q[0]<=nr_thres && q[2]-p[12]>=thres))) )//2011/0607 start
            {
                pix_out[0] = minavg;
            }
        }       
    }
    else if (is_color == 2) {
        //DPC
        AVGPIX[0]=(p[11]+p[13])>>1;//RL
        AVGPIX[1]=(p[7]+p[17])>>1;//UD
        AVGPIX[2]=(p[0]+p[24])>>1;//D1
        AVGPIX[3]=(p[4]+p[20])>>1;//D2
        
        //DIFPIX[0]=abs(p[10]-p[14]);//RL
        //DIFPIX[1]=abs(p[2]-p[22]);//UD
        //DIFPIX[2]=abs(p[0]-p[24]);//D1
        //DIFPIX[3]=abs(p[4]-p[20]);//D2
        
        minavg=MIN(MIN(AVGPIX[0], AVGPIX[1]), MIN(AVGPIX[2], AVGPIX[3])); 
        maxavg=MAX(MAX(AVGPIX[0], AVGPIX[1]), MAX(AVGPIX[2], AVGPIX[3])); 
                                
        q[0]=p[2];
        q[1]=p[7];
        q[2]=p[10];
        q[3]=p[11];
        q[4]=p[12];
        q[5]=p[13];
        q[6]=p[14];
        q[7]=p[17];
        q[8]=p[22];
        
        SortArray(q, 9);
        pix_out[0] = p[12];
        //if( ((p[12]-maxavg>=thres) && (q[8]-p[12]<=nr_thres && p[12]-q[7]>=thres)) )//2011/0607 start
        //{
        //    pix_out[0] = maxavg;
        //}
        //else if( (minavg-p[12]>=DpcThd2) && (q[0]==p[12] && q[1]-p[12]>=DpcThd2) )
        if( ((minavg-p[12]>=thres/2) && (p[12]-q[0]<=nr_thres/2 && q[1]-p[12]>=thres/2)) )//2011/0607 start
        {
            pix_out[0] = minavg;
        } 
    }
    else if (is_color == 3) {
        //DPC
        AVGPIX[0]=(p[10]+p[14])>>1;//RL
        AVGPIX[1]=(p[2]+p[22])>>1;//UD
        AVGPIX[2]=(p[0]+p[24])>>1;//D1
        AVGPIX[3]=(p[4]+p[20])>>1;//D2
        
        //DIFPIX[0]=abs(p[10]-p[14]);//RL
        //DIFPIX[1]=abs(p[2]-p[22]);//UD
        //DIFPIX[2]=abs(p[0]-p[24]);//D1
        //DIFPIX[3]=abs(p[4]-p[20]);//D2
        
        minavg=MIN(MIN(AVGPIX[0], AVGPIX[1]), MIN(AVGPIX[2], AVGPIX[3])); 
        maxavg=MAX(MAX(AVGPIX[0], AVGPIX[1]), MAX(AVGPIX[2], AVGPIX[3])); 
                                
        q[0]=p[0];
        q[1]=p[2];
        q[2]=p[4];
        q[3]=p[10];
        q[4]=p[12];
        q[5]=p[14];
        q[6]=p[20];
        q[7]=p[22];
        q[8]=p[24];
        
        SortArray(q, 9);
        pix_out[0] = p[12];
        //if( ((p[12]-maxavg>=thres) && (q[8]-p[12]<=nr_thres && p[12]-q[7]>=thres)) )//2011/0607 start
        //{
        //    pix_out[0] = maxavg;
        //}
        //else if( (minavg-p[12]>=DpcThd2) && (q[0]==p[12] && q[1]-p[12]>=DpcThd2) )
        if( ((minavg-p[12]>=thres) && (p[12]-q[0]<=nr_thres && q[1]-p[12]>=thres)) )//2011/0607 start
        {
            pix_out[0] = minavg;
        }
    }
   
}
extern "C" __global__
void dpc(const short* src, short* dst,int width, int height,int pad_w,int pad_h,int thres,int bayer_pattern) {

    int row = (blockIdx.y * blockDim.y + threadIdx.y)*2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int i_width = width + pad_w;
    int i_height = height + pad_h;
    int pad_w2 = pad_w/2;
    int pad_h2 = pad_h/2;
    int offset = (row+pad_h2)*i_width + col + pad_w2;
    int shift_offset = (row)*width + col;
    int thres1 = 50;
    int thres2 = 100;
    if ((row < i_height) && (col < i_width) && row>=0 && col>=0) {
        short pix_out[3];
        DPC_kernel(src,1,row+2,col+2,i_width,thres1,thres2,pix_out);
        dst[shift_offset] = pix_out[0];
        shift_offset++;
        offset++;
        DPC_kernel(src,2,row+2,col+2+1,i_width,thres1,thres2,pix_out);
        dst[shift_offset] = pix_out[0];
        shift_offset+=width-1;
        offset+=i_width-1;
        DPC_kernel(src,0,row+2+1,col+2,i_width,thres1,thres2,pix_out);
        dst[shift_offset] = pix_out[0];
        shift_offset++;
        offset++;
        DPC_kernel(src,1,row+2+1,col+2+1,i_width,thres1,thres2,pix_out);
        dst[shift_offset] = pix_out[0];
    }
    /*
    if ((row < height) && (col < width)) {
        int offset = row*src_width + col;
        int AVGPIX[4];
        int minavg;
        int maxavg;
        short p[25]; //5x5 crop_image
        int mx,mn,lu,ru,ld,rd;
        int q[13];
        for(int i=0;i<5;i++) {
            for(int j=0;j<5;j++) {
                int offset = (row+i-2)*src_width + col + j -2;
                p[i*5+j] = src[offset];
            }
        }
        //DPC
        int INMAX, INMIN, OUTMAX, OUTMIN;
        INMAX=MAX(MAX(p[6], p[8]), MAX(p[16], p[18]));
        INMIN=MIN(MIN(p[6], p[8]), MIN(p[16], p[18]));
        OUTMAX=MAX(MAX(p[0], p[4]), MAX(p[20], p[24]));
        OUTMIN=MIN(MIN(p[0], p[4]), MIN(p[20], p[24]));
        if(abs(p[12]-INMAX)<=thres || abs(INMIN-p[12])<=thres)
        {
            AVGPIX[0]=(p[10]+p[14])>>1;//RL
            AVGPIX[1]=(p[2]+p[22])>>1;//UD
            AVGPIX[2]=(p[0]+p[24])>>1;//D1
            AVGPIX[3]=(p[4]+p[20])>>1;//D2					
        }
        else
        {
            AVGPIX[0]=(p[10]+p[14])>>1;//RL
            AVGPIX[1]=(p[2]+p[22])>>1;//UD
            AVGPIX[2]=(p[6]+p[18])>>1;//D1
            AVGPIX[3]=(p[8]+p[16])>>1;//D2												
        }
        minavg=MIN(MIN(AVGPIX[0], AVGPIX[1]), MIN(AVGPIX[2], AVGPIX[3])); 
        maxavg=MAX(MAX(AVGPIX[0], AVGPIX[1]), MAX(AVGPIX[2], AVGPIX[3])); 
        if(1) {
            //0614 modify		
            
            q[0]=p[2];
            q[1]=p[6];
            q[2]=p[8];
            q[3]=p[10];
            q[4]=p[12];
            q[5]=p[14];
            q[6]=p[16];
            q[7]=p[18];
            q[8]=p[22];
                
            SortArray(q, 9);
                
            lu=(2*p[6]-p[0]);
            ru=(2*p[16]-p[20]);
            ld=(2*p[18]-p[24]);
            rd=(2*p[8]-p[4]);

            mx=MAX(MAX(lu, ru), MAX(ld, rd));
            mn=MIN(MIN(lu, ru), MIN(ld, rd));
            dst[row*width + col] = src[offset+src_width*2+2];
            
                
                //if( (p[12]-maxavg>=DpcThd2) && (q[8]-p[12]<=DpcNRThd && p[12]-q[6]>=DpcThd2) )
            //if( ((p[12]-maxavg>=thres && p[12]>mx) || ((p[12]-maxavg>=thres) && (q[8]-p[12]<=thres && p[12]-q[6]>=thres))) )//2011/0607 start
            //{
            //    dst[row*width + col] = maxavg;
            //}					
            //else if( (minavg-p[12]>=DpcThd2) && (p[12]-q[0]<=DpcNRThd && q[2]-p[12]>=DpcThd2) )
            if( ((minavg-p[12]>=thres && p[12]<mn) || ((minavg-p[12]>=thres) && (p[12]-q[0]<=thres && q[2]-p[12]>=thres))) )//2011/0607 start
            {
                dst[row*width + col] = minavg;
            }        
        }
        if(0)
        {
            //Local Peak				
            lu=(2*p[6]-p[0]);
            ru=(2*p[16]-p[20]);
            ld=(2*p[18]-p[24]);
            rd=(2*p[8]-p[4]);
            
            mx=MAX(MAX(lu, ru), MAX(ld, rd));
            mn=MIN(MIN(lu, ru), MIN(ld, rd));
            offset = row*src_width + col;
            if( (p[12]-maxavg>=thres/2 && p[12]>mx) )
            {					
                dst[row*width + col]=maxavg;					
            }
            else if( (minavg-p[12]>=thres/2 && p[12]<mn) )
            {					
                dst[row*width + col]=minavg;					
            }
            else {
                dst[row*width + col] = src[offset+src_width*2+2];
            }
        } */
        //offset = row*src_width + col;
        //dst[row*width + col] = src[offset+src_width*2+2];
        //z[offset] = p[offset];
        /*
        int p0 = src[offset+src_width*2+2];
        int p1 = src[offset];
        int p2 = src[offset+2];
        int p3 = src[offset+4];
        int p4 = src[offset+src_width*2];
        int p5 = src[offset+src_width*2+4];
        int p6 = src[offset+src_width*4];
        int p7 = src[offset+src_width*4+2];
        int p8 = src[offset+src_width*4+4];
        if ((abs(p1 - p0) > thres) && (abs(p2 - p0) > thres) && (abs(p3 - p0) > thres) && (abs(p4 - p0) > thres) &&
        (abs(p5 - p0) > thres) && (abs(p6 - p0) > thres) && (abs(p7 - p0) > thres) && (abs(p8 - p0) > thres))
        {
                int dv = abs(2 * p0 - p2 - p7);
                int dh = abs(2 * p0 - p4 - p5);
                int ddl = abs(2 * p0 - p1 - p8);
                int ddr = abs(2 * p0 - p3 - p6);
                if (dv <= dh && dv <= ddl && dv <= ddr)
                    p0 = (p2 + p7 + 1) / 2;
                else if (dh <= dv && dh <= ddl && dh <= ddr)
                    p0 = (p4 + p5 + 1) / 2;
                else if (ddl <= dv && ddl <= dh && ddl <= ddr)
                    p0 = (p1 + p8 + 1) / 2;
                else 
                    p0 = (p3 + p6 + 1) / 2;
        }
        offset = row*width + col;
        dst[offset] = p0;
        
    }*/
 }