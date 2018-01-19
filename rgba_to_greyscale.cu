
#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int threadsAbove  = (blockDim.y * blockIdx.y) + threadIdx.y;
  int threadsToLeft = (blockDim.x * blockIdx.x) + threadIdx.x;

  int numThreadsPerRowOfCompleteGridWidth = gridDim.x * blockDim.x;
  int currentThreadOffset = threadsToLeft + threadsAbove * numThreadsPerRowOfCompleteGridWidth;

  if(currentThreadOffset >= (numRows * numCols))
    return;

  greyImage[currentThreadOffset] =
    .299f * rgbaImage[currentThreadOffset].x +
    .587f * rgbaImage[currentThreadOffset].y +
    .114f * rgbaImage[currentThreadOffset].z;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const int NUM_THREAD_X = 32;
  const int NUM_THREAD_Y = 32;
  const dim3 blockSize( NUM_THREAD_X, NUM_THREAD_Y, 1 );

  const int NUM_BLOCKS_IN_X_DIR = ceil(numCols/(float) NUM_THREAD_X);
  const int NUM_BLOCKS_IN_Y_DIR = ceil(numRows/(float) NUM_THREAD_Y);
  const dim3 gridSize(NUM_BLOCKS_IN_X_DIR, NUM_BLOCKS_IN_Y_DIR, 1);

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
