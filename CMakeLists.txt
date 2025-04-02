cmake_minimum_required(VERSION 3.18)
project(SquareArray LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA is available. Building with CUDA support.")

    find_package(CUDAToolkit REQUIRED)

    add_library(square_array SHARED square_array.cu)
    target_link_libraries(square_array PRIVATE CUDA::cudart)

    add_executable(main_square_array main_cuda.cpp)
    target_include_directories(main_square_array PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Explicitly link CUDA runtime to the executable
    target_link_libraries(main_square_array PRIVATE square_array CUDA::cudart)
else()
    message(STATUS "CUDA not found. Falling back to OpenMP.")

    find_package(OpenMP REQUIRED)
    add_library(square_array SHARED square_array.cpp)
    target_link_libraries(square_array PUBLIC OpenMP::OpenMP_CXX)

    add_executable(main_square_array main_no_cuda.cpp)
endif()

target_link_libraries(main_square_array PRIVATE square_array)


install(TARGETS square_array main_square_array
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(FILES square_array.h DESTINATION include)


