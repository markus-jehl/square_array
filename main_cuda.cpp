#include "square_array.h"
#include <iostream>
#include <cuda_runtime.h>

void print_array(const char* label, float* array, size_t size) {
    std::cout << label << ": ";
    for (size_t i = 0; i < size; ++i)
        std::cout << array[i] << " ";
    std::cout << "\n";
}

int main() {
    const size_t N = 10;
    float sum = 0.0f;

    // Host array
    float host_data[N];
    for (size_t i = 0; i < N; ++i)
        host_data[i] = i + 1;

    std::cout << "\n";
    print_array("Host input", host_data, N);

    square_array(host_data, N, &sum);
    print_array("Host output", host_data, N);
    std::cout << "Host sum of squares: " << sum << "\n";

    // Device array
    sum = 0.0f;
    float temp[N];
    for (size_t i = 0; i < N; ++i)
        temp[i] = i + 2;
    float* dev_data;
    cudaMalloc(&dev_data, N * sizeof(float));
    cudaMemcpy(dev_data, temp, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\n";
    print_array("Device input", temp, N);

    square_array(dev_data, N, &sum);
    cudaMemcpy(temp, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Device output", temp, N);
    std::cout << "Device sum of squares: " << sum << "\n";
    cudaFree(dev_data);

    // Managed memory
    sum = 0.0f;
    float* managed;
    cudaMallocManaged(&managed, N * sizeof(float));
    for (size_t i = 0; i < N; ++i)
        managed[i] = i + 3;

    std::cout << "\n";
    print_array("Managed input", managed, N);

    square_array(managed, N, &sum);
    cudaDeviceSynchronize();
    print_array("Managed output", managed, N);
    std::cout << "Managed sum of squares: " << sum << "\n";
    cudaFree(managed);

    return 0;
}

