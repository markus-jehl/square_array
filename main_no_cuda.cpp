#include "square_array.h"
#include <iostream>

void print_array(const char* label, const float* array, size_t size) {
    std::cout << label << ": ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const size_t N = 10;
    float* data = new float[N];

    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    print_array("Host before", data, N);
    square_array(data, N);  // This will use the OpenMP version
    print_array("Host after ", data, N);

    delete[] data;
    return 0;
}

