#include "square_array.h"
#include <omp.h>

void square_array(float* array, size_t size) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        array[i] *= array[i];
    }
}

