extern "C" __global__
void nv12_disparity_kernel(
    const unsigned char* srcY,
    short* disparity,
    int W,
    int H,
    int maxDisparity,
    float depthStrength,
    float edgeWeight,
    float lumaWeight,
    float verticalWeight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    const int inRow = y * W;
    const int xLeftNeighbor = (x > 0) ? (x - 1) : 0;
    const int yUpNeighbor = (y > 0) ? (y - 1) : 0;

    const int luma = (int)srcY[inRow + x];
    const int left = (int)srcY[inRow + xLeftNeighbor];
    const int up = (int)srcY[yUpNeighbor * W + x];

    float edge = (float)(abs(luma - left) + abs(luma - up)) * (1.0f / 510.0f);
    if (edge < 0.0f) edge = 0.0f;
    if (edge > 1.0f) edge = 1.0f;

    const float lumaNorm = (float)luma * (1.0f / 255.0f);
    const float vertical = (H > 1) ? ((float)y / (float)(H - 1)) : 0.0f;

    float score = ((edge * edgeWeight) + (lumaNorm * lumaWeight) + (vertical * verticalWeight)) * depthStrength;
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;

    int d = (int)(score * (float)maxDisparity + 0.5f);
    if (d < 0) d = 0;
    if (d > maxDisparity) d = maxDisparity;
    disparity[inRow + x] = (short)d;
}

extern "C" __global__
void nv12_sbs_y_kernel(
    const unsigned char* srcY,
    const short* disparity,
    unsigned char* dstY,
    int W,
    int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    const int inRow = y * W;
    const int outRow = y * (W * 2);
    const int d = (int)disparity[inRow + x];

    int srcLeft = x - d;
    int srcRight = x + d;
    if (srcLeft < 0) srcLeft = 0;
    if (srcRight >= W) srcRight = W - 1;

    dstY[outRow + x] = srcY[inRow + srcLeft];
    dstY[outRow + W + x] = srcY[inRow + srcRight];
}

extern "C" __global__
void nv12_sbs_uv_kernel(
    const unsigned char* srcUV,
    const short* disparity,
    unsigned char* dstUV,
    int W,
    int H)
{
    const int chromaX = blockIdx.x * blockDim.x + threadIdx.x;
    const int chromaY = blockIdx.y * blockDim.y + threadIdx.y;
    const int chromaW = W / 2;
    const int chromaH = H / 2;
    if (chromaX >= chromaW || chromaY >= chromaH) {
        return;
    }

    const int srcYRow = (chromaY * 2) * W;
    const int srcX = chromaX * 2;
    const int disparityUv = ((int)disparity[srcYRow + srcX]) / 2;

    int srcLeftPair = chromaX - disparityUv;
    int srcRightPair = chromaX + disparityUv;
    if (srcLeftPair < 0) srcLeftPair = 0;
    if (srcRightPair >= chromaW) srcRightPair = chromaW - 1;

    const int srcUvRow = chromaY * W;
    const int dstUvRow = chromaY * (W * 2);

    const int leftSrc = srcUvRow + (srcLeftPair * 2);
    const int rightSrc = srcUvRow + (srcRightPair * 2);
    const int leftDst = dstUvRow + (chromaX * 2);
    const int rightDst = dstUvRow + W + (chromaX * 2);

    dstUV[leftDst] = srcUV[leftSrc];
    dstUV[leftDst + 1] = srcUV[leftSrc + 1];
    dstUV[rightDst] = srcUV[rightSrc];
    dstUV[rightDst + 1] = srcUV[rightSrc + 1];
}
