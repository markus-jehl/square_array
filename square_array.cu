#include "square_array.h"
#include "square_op.h"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void square_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = square_compute(data[idx]);
    }
}

void square_array(float* array, size_t size) {
    float* device_array = nullptr;
    bool needs_copy_back = false;

    // Detect pointer type
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, array);
    bool is_device_ptr = false;

    if (err == cudaSuccess) {
#if CUDART_VERSION >= 10000
        is_device_ptr = (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
        is_device_ptr = (attr.memoryType == cudaMemoryTypeDevice);
#endif
    }

    if (is_device_ptr) {
        printf("input array is already on device\n");
        device_array = array;
    } else {
        // Allocate device memory and copy data
        printf("copying input array to device\n");
        cudaMalloc(&device_array, size * sizeof(float));
        cudaMemcpy(device_array, array, size * sizeof(float), cudaMemcpyHostToDevice);
        needs_copy_back = true;
    }

    // Launch kernel
    int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    square_kernel<<<blocks, threads>>>(device_array, size);
    cudaDeviceSynchronize();

    // Copy result back if we allocated temp device memory
    if (needs_copy_back) {
        printf("copying result back to host\n");
        cudaMemcpy(array, device_array, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_array);
    }
}

