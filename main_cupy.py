import cupy as cp
import ctypes
import os

# Load shared library
lib = ctypes.CDLL(os.path.abspath("build/libsquare_array.so"))

# Define argument types
lib.square_array.argtypes = [
    ctypes.c_void_p,          # float* array (on device)
    ctypes.c_size_t,          # size_t size
    ctypes.c_void_p           # float* result_sum (on device)
]
lib.square_array.restype = None

# Create a CuPy array on the GPU
arr = cp.arange(1, 11, dtype=cp.float32)
sum_gpu = cp.zeros(1, dtype=cp.float32)

print("Before:", arr)

# Call the CUDA C function
lib.square_array(
    arr.data.ptr,
    arr.size,
    sum_gpu.data.ptr
)

# Synchronize to ensure results are ready
cp.cuda.Device().synchronize()

print("After: ", arr)
print("Sum of squares (from GPU):", float(sum_gpu[0]))

