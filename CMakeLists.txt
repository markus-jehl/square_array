cmake_minimum_required(VERSION 3.18)
project(SquareArray LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA is available. Building with CUDA support.")

    find_package(CUDAToolkit REQUIRED)

    add_library(square_array square_array.cu)
    target_link_libraries(square_array PRIVATE CUDA::cudart)

    add_executable(main main_cuda.cpp)
    target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS}) # ? Add this line
else()
    message(STATUS "CUDA not found. Falling back to OpenMP.")

    find_package(OpenMP REQUIRED)
    add_library(square_array square_array.cpp)
    target_link_libraries(square_array PUBLIC OpenMP::OpenMP_CXX)

    add_executable(main main_no_cuda.cpp)
endif()

target_link_libraries(main PRIVATE square_array)

