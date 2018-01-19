#include <stdio.h>
#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  int whole_block_y_sums = blockIdx.y * blockDim.y;
  int rem_y = threadIdx.y;
  int above = whole_block_y_sums + rem_y;
  int r = above;

  int whole_block_x_sums  = blockIdx.x * blockDim.x;
  int rem_x = threadIdx.x;
  int left = whole_block_x_sums + rem_x;
  int c = left;

  if( r >= numRows || c >= numCols)
    return;

  int offset_1D = (r * numCols) + c;
  int mr = static_cast<int>(numRows - 1), mc =  static_cast<int>(numCols - 1);
  int half_fw = (filterWidth / 2);
  float result = 0.f;

  for (int filter_r = -half_fw; filter_r <= half_fw; ++filter_r) {
    for (int filter_c = -half_fw; filter_c <= half_fw; ++filter_c) {

      int rfr = (r + filter_r);
      int image_r = rfr < 0 ? 0 : rfr;
      image_r = image_r < numRows ? image_r : (numRows - 1);

      int cfc = (c + filter_c);
      int image_c = cfc < 0 ? 0 : cfc;
      image_c = image_c < numCols ? image_c : (numCols-1);

      float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
      float filter_value = filter[(filter_r + half_fw) * filterWidth + filter_c + half_fw];

      result += image_value * filter_value;
    }
  }

  outputChannel[offset_1D] = result;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int threadsAbove  = (blockDim.y * blockIdx.y) + threadIdx.y;
  int threadsToLeft = (blockDim.x * blockIdx.x) + threadIdx.x;

  int numThreadsPerRowOfCompleteGridWidth = gridDim.x * blockDim.x;
  int currentThreadOffset = threadsToLeft + threadsAbove * numThreadsPerRowOfCompleteGridWidth;

  if(currentThreadOffset >= (numRows * numCols))
    return;

  redChannel[currentThreadOffset] = inputImageRGBA[currentThreadOffset].x;
  greenChannel[currentThreadOffset] = inputImageRGBA[currentThreadOffset].y;
  blueChannel[currentThreadOffset] = inputImageRGBA[currentThreadOffset].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  size_t fsz = sizeof(float) * (filterWidth*filterWidth); // filter size
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, fsz, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{

  printf("filterWidth %d numRows %d numCols %d \n",filterWidth, numRows, numCols);

  const int NUM_THREAD_X = 16; const int NUM_THREAD_Y = 16;
  const int NUM_BLOCKS_IN_X_DIR = ceil(numCols/(float) NUM_THREAD_X);
  const int NUM_BLOCKS_IN_Y_DIR = ceil(numRows/(float) NUM_THREAD_Y);

  const dim3 blockSize ( NUM_THREAD_X, NUM_THREAD_Y, 1 );
  const dim3 gridSize (NUM_BLOCKS_IN_X_DIR, NUM_BLOCKS_IN_Y_DIR, 1);

  separateChannels<<<gridSize, blockSize>>>
    (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //////////////////////////////////////////////////////////////////
  gaussian_blur<<<gridSize, blockSize>>>
    (d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize>>>
    (d_green, d_greenBlurred,numRows,numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize>>>
    (d_blue, d_blueBlurred,numRows,numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //////////////////////////////////////////////////////////////////

  recombineChannels<<<gridSize, blockSize>>>
    (d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
