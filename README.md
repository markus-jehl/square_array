# SquareArray

**SquareArray** is a minimal C++/CUDA project that performs elementwise squaring of 1D float arrays.  
It automatically uses CUDA for GPU acceleration when available, and falls back to OpenMP for parallel CPU execution when CUDA is not detected.  
The project uses modern CMake to configure the appropriate backend at build time.

## Features

- GPU acceleration with CUDA (if available)
- CPU parallelization using OpenMP
- Automatic backend selection with CMake
- Examples of host, device, and managed memory usage

## Requirements

- CMake >= 3.18
- C++ compiler (GCC, Clang, or MSVC)
- Optional:
  - CUDA Toolkit (from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads))
  - OpenMP support (usually available with most C++ compilers)

## Build Instructions

### 1. Clone the repository

```bash
cd square_array_project
mkdir build
cd build
cmake ..
cmake --build .
./main_square_array
```

```bash
# run python wrapper using different array backends and devices
# make sure that numpy and array-api-compat are installed
# cupy and pytorch and array-api-strict are optional
python main.py
```

