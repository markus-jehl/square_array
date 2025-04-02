#pragma once
#include "cuda_compat.h"
#include "atomic_sum.h"

CUDA_HOST_DEVICE inline void compute_and_accumulate(float* array, size_t idx, float* result_sum) {
    array[idx] *= array[idx];
    atomic_sum(result_sum, array[idx]);
}

