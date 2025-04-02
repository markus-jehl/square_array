import numpy as xp
import ctypes
import os

# Load the compiled shared library
lib = ctypes.CDLL(os.path.abspath("build/libsquare_array.so"))
lib.square_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
lib.square_array.restype = None

# Create NumPy array
data = xp.arange(1, 11, dtype=xp.float32)
print("Before:", data)

# Create ctypes pointer to sum
sum_out = ctypes.c_float()

# Call the function
lib.square_array(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 ctypes.c_size_t(len(data)),
                 ctypes.byref(sum_out))

# Output results
print("After: ", data)
print("Sum of squares:", sum_out.value)

