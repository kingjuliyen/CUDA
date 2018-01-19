
#include <stdio.h>
#include <float.h>
#include <algorithm>
#include "utils.h"
#include "blelloch_scan.cu"

#define DEF_CUDA_KERN __global__ void
#define DEF_CUDA_DEV_FN __device__
#define EXTERN_TH_LOCAL_SHMEM extern __shared__
#define SYNC_THREADS __syncthreads

#define DFLT_TH_PER_BLK 1024

enum ROPTYPE {MAXFN, MINFN};
typedef __device__ float (*REDUCE_OP) (float a, float b);
DEF_CUDA_DEV_FN float getmax(float a, float b) {  return (a>b) ? a : b; }
DEF_CUDA_DEV_FN float getmin(float a, float b) {  return (a<b) ? a : b; }


DEF_CUDA_KERN reduce_kernel( const float* const d_in_data,
                              float *out, size_t N, ROPTYPE rop_ty )
{
  EXTERN_TH_LOCAL_SHMEM float sh[];
  size_t th_glbl_ofst = (blockDim.x * blockIdx.x) + threadIdx.x;
  int thid = threadIdx.x;

  REDUCE_OP rop = (rop_ty == MINFN) ? getmin : getmax;
  const float IDENTITY = (rop_ty == MINFN) ? FLT_MAX : FLT_MIN;

  sh[thid] = (th_glbl_ofst < N) ? d_in_data[th_glbl_ofst] : IDENTITY;
  SYNC_THREADS();

  for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)  {
    if(thid < s) {
      sh[thid] = rop(sh[thid], sh[thid + s]);
    }
    SYNC_THREADS();
  }
  if(thid == 0) {
    out[blockIdx.x] = sh[thid];
  }
}

void find_min_max ( const float* const d_in_data,
                    float &min_logLum, float &max_logLum,
                    const size_t numRows, const size_t numCols,
                    const int THREADS_PER_BLOCK = DFLT_TH_PER_BLK )
{
  int N = numRows * numCols;
  {
    // #define CROSS_CHECK
    #ifdef CROSS_CHECK
    float * h_t = (float *) malloc (sizeof(float) * N);
    checkCudaErrors(cudaMemcpy(h_t, d_in_data,   sizeof(float) * N, cudaMemcpyDeviceToHost));
    min_logLum = *std::min_element(h_t, h_t+N);
    max_logLum = *std::max_element(h_t, h_t+N);
    printf("cross_check min_logLum %f max_logLum %f \n", min_logLum, max_logLum);
    free(h_t);
    #endif
  }


  int num_blocks = ceil(N/THREADS_PER_BLOCK);
  size_t shm_sz = THREADS_PER_BLOCK * sizeof(float);

  float * d_out = 0;
  checkCudaErrors(cudaMalloc(&d_out,   sizeof(float) * num_blocks));
  float * h_out = (float *) malloc (sizeof(float) * num_blocks);

  reduce_kernel <<< num_blocks, THREADS_PER_BLOCK, shm_sz >>> (d_in_data, d_out, N, MINFN);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(h_out, d_out,   sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));
  min_logLum = *std::min_element(h_out, h_out+num_blocks) ;

  reduce_kernel <<< num_blocks, THREADS_PER_BLOCK, shm_sz >>> (d_in_data, d_out, N, MAXFN);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(h_out, d_out,   sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));
  max_logLum = *std::max_element(h_out, h_out+num_blocks) ;

  printf("min_logLum %f max_logLum %f range %f \n", min_logLum, max_logLum, (max_logLum - min_logLum));

  free(h_out);
  checkCudaErrors(cudaFree(d_out));
}

DEF_CUDA_KERN gen_histo_kernel( const float* const d_logLuminance, int N,
                                unsigned int * d_histo, float min_logLum,
                                float logLumRange, const size_t numBins)
{
  size_t th_glbl_ofst = (blockDim.x * blockIdx.x) + threadIdx.x;
  if(th_glbl_ofst >= N)
    return;

  assert(th_glbl_ofst >= 0 && th_glbl_ofst < N);

  int x = static_cast<unsigned int>(numBins - 1);
  int y = static_cast<unsigned int>
    ((d_logLuminance[th_glbl_ofst] - min_logLum) / logLumRange * numBins);

  int bin = (x < y) ? x : y;
  assert(bin >=0 && bin < numBins);
  atomicAdd(d_histo + bin, 1);
}

unsigned int *
  gen_histogram(  const float* const d_logLuminance,
                  const size_t numRows, const size_t numCols,
                  float &min_logLum, float &logLumRange, const size_t numBins,
                  const int THREADS_PER_BLOCK = DFLT_TH_PER_BLK  )
{
  printf("numBins %d \n", numBins);

  int N = numRows * numCols;
  int num_blocks = ceil(N/THREADS_PER_BLOCK);
  unsigned int * d_histo;
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  gen_histo_kernel <<< num_blocks, THREADS_PER_BLOCK >>> (d_logLuminance, N, d_histo, min_logLum, logLumRange, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  return d_histo;
}


void gen_cdf(unsigned int * d_histo, unsigned int * d_cdf, const size_t numBins)
{
  blelloch_scan_test();
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  find_min_max(d_logLuminance, min_logLum, max_logLum, numRows, numCols);
  float logLumRange = max_logLum - min_logLum;
  printf("logLumRange %f \n", logLumRange);


  unsigned int * d_histo = gen_histogram(d_logLuminance, numRows, numCols, min_logLum, logLumRange, numBins);
  {
    // #define HIST_CHECK
    #ifdef HIST_CHECK
    unsigned int *  h_t = (unsigned int * ) malloc (sizeof(unsigned int) * numBins);
    checkCudaErrors(cudaMemcpy(h_t, d_histo,   sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
    int acc = 0;
    for(int k=0; k<numBins; k++) {
      printf("k %d h_t[k] %d \n", k, h_t[k]);
      acc += h_t[k];
    }
    printf("acc %d \n", acc);
    free(h_t);
    #endif
  }
  gen_cdf(d_histo, d_cdf, numBins);
  checkCudaErrors(cudaFree(d_histo));
}
