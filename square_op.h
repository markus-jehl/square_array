#pragma once
#include "cuda_compat.h"

// Shared logic as a single inline function
CUDA_HOST_DEVICE inline float square_compute(float x) {
    return x * x * x;
}

