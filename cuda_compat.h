#pragma once

// If compiling with NVCC or a CUDA-aware compiler
#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_DEVICE __device__
  #define CUDA_GLOBAL __global__
#else
  // Define as nothing on non-CUDA systems
  #define CUDA_HOST_DEVICE
  #define CUDA_DEVICE
  #define CUDA_GLOBAL
#endif

