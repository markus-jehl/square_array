# SquareArray

**SquareArray** is a minimal C++/CUDA project that performs elementwise squaring of 1D float arrays.  
It automatically uses CUDA for GPU acceleration when available, and falls back to OpenMP for parallel CPU execution when CUDA is not detected.  
When CUDA is availalbe, host or device arrays can be directly passed to the library function
- [see this example](main_cuda.cpp#L40)

## Features

- CPU parallelization using OpenMP
- GPU acceleration with CUDA (if available)
- Automatic backend selection with CMake
- python wrapper for numpy, cupy, pytorch arrays

## Requirements

- CMake >= 3.18
- C++ compiler (GCC, Clang, or MSVC)
- Optional:
  - CUDA Toolkit (from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads))

## Build + run examples

The build work on systems with and without CUDA.
The main executable is build from [main_cuda.cpp](main_cuda.cpp) or [main_no_cuda.cpp](main_no_cuda.cpp) - depending on the availability of CUDA - and demonstrates use cases with different array types.

```bash
mkdir build
cd build
cmake ..
cmake --build .
./main
```

We also provide a python wrapper that can be used with different array types on different devices.

```bash
# run python wrapper using different array backends and devices
# make sure that numpy and array-api-compat are installed
# cupy and pytorch and array-api-strict are optional
python main.py
```

