#include "square_array.h"
#include <iostream>
#include <cuda_runtime.h>

void print_array(const char* label, const float* array, size_t size) {
    std::cout << label << ": ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const size_t N = 10;

    // --- Host memory
    float* host_array = new float[N];
    for (size_t i = 0; i < N; ++i) host_array[i] = static_cast<float>(i + 1);
    print_array("Host before", host_array, N);
    square_array(host_array, N);  // Kernel will detect and handle host pointer
    print_array("Host after ", host_array, N);
    delete[] host_array;

    // --- Device memory
    float* device_array;
    float temp[N];
    for (size_t i = 0; i < N; ++i) temp[i] = static_cast<float>(i + 2);
    cudaMalloc(&device_array, N * sizeof(float));
    cudaMemcpy(device_array, temp, N * sizeof(float), cudaMemcpyHostToDevice);
    square_array(device_array, N);  // In-place device processing
    cudaMemcpy(temp, device_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Device after", temp, N);
    cudaFree(device_array);

    // --- Managed memory
    float* managed_array;
    cudaMallocManaged(&managed_array, N * sizeof(float));
    for (size_t i = 0; i < N; ++i) managed_array[i] = static_cast<float>(i + 3);
    square_array(managed_array, N);  // In-place managed processing
    cudaDeviceSynchronize();  // Ensure kernel completion
    print_array("Managed after", managed_array, N);
    cudaFree(managed_array);

    return 0;
}

