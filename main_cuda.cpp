#include "square_array.h"
#include <iostream>
#include <cuda_runtime.h>

void print_array(const char* label, float* array, size_t size) {
    std::cout << label << ": ";
    // print max 10 elements
    size_t print_size = (size > 10) ? 10 : size;
    for (size_t i = 0; i < print_size; ++i)
        std::cout << array[i] << " ";
    // print ellipses if size > 10 and the last element
    if (size > 10)
        std::cout << "... " << array[size - 1];
    std::cout << "\n";
}

int main() {
    const size_t N = 10000;
    float sum = 0.0f;

    ////////////////////////////////////////////////////////
    // Host array use case
    ////////////////////////////////////////////////////////

    float host_data[N];
    for (size_t i = 0; i < N; ++i)
        host_data[i] = i + 1;

    std::cout << "\n";
    print_array("Host input", host_data, N);

    try { 
        // use invalid device ID 999 for testing
        square_array(host_data, N, &sum, 999);
        print_array("Host output", host_data, N);
        std::cout << "Host sum of squares: " << sum << "\n";
    }
    catch (const std::runtime_error& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
    }
    
    // use default device ID 0
    square_array(host_data, N, &sum);
    print_array("Host output", host_data, N);
    std::cout << "Host output sum: " << sum << "\n";

    ////////////////////////////////////////////////////////
    // Device array use case
    ////////////////////////////////////////////////////////

    float sum_d = 0.0f;
    float temp[N];
    for (size_t i = 0; i < N; ++i)
        temp[i] = i + 1;
    float* dev_data;
    cudaMalloc(&dev_data, N * sizeof(float));
    cudaMemcpy(dev_data, temp, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\n";
    print_array("Device input", temp, N);

    square_array(dev_data, N, &sum_d, 0);
    cudaMemcpy(temp, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Device output", temp, N);
    std::cout << "Device output sum: " << sum_d << "\n";
    cudaFree(dev_data);

    ////////////////////////////////////////////////////////
    // CUDA memory managed use case
    ////////////////////////////////////////////////////////

    // get the number of cuda devices - because we want to run on the last device
    int device_count;
    cudaGetDeviceCount(&device_count);

    float sum_m = 0.0f;
    float* managed;
    cudaMallocManaged(&managed, N * sizeof(float));
    for (size_t i = 0; i < N; ++i)
        managed[i] = i + 1;

    std::cout << "\n";
    print_array("Managed input", managed, N);

    // use cuda managed memory and run on last device
    square_array(managed, N, &sum_m, device_count - 1);
    cudaDeviceSynchronize();
    print_array("Managed output", managed, N);
    std::cout << "Managed output sum: " << sum_m << "\n";
    cudaFree(managed);

    return 0;
}

