extern "C" __global__
void dpc(const short* src, short* dst,int width, int height,int pad_w,int pad_h,int thres) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int src_width = width+pad_w*2;
    int src_height = height+pad_h*2;
    if ((row < height) && (col < width)) {
        int offset = row*src_width + col;
        //z[offset] = p[offset];

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
    }
 }