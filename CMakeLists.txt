cmake_minimum_required(VERSION 3.18)
project(SquareArray LANGUAGES C CXX)

# derive the project version using git describe / the name of the tarball / zip

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
  message(STATUS "No .git directory found ? trying to parse version from directory name")

  # Extract from directory name: square_array-1.2.3 or v1.2.3
  get_filename_component(FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME)
  string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+)" TARBALL_VERSION ${FOLDER_NAME})
  if(TARBALL_VERSION)
    set(PROJECT_VERSION "${TARBALL_VERSION}")
  else()
    set(PROJECT_VERSION "0.0.0")
  endif()
else()
  execute_process(
    COMMAND git describe --tags --long --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Git describe: ${GIT_DESCRIBE}")

  # Extract base version (remove v prefix)
  string(REGEX MATCH "v?([0-9]+\\.[0-9]+\\.[0-9]+)" PROJECT_VERSION ${GIT_DESCRIBE})
  string(REGEX REPLACE "^v" "" PROJECT_VERSION "${PROJECT_VERSION}")

  # Parse version components from PROJECT_VERSION
  string(REGEX MATCHALL "[0-9]+" VERSION_PARTS "${PROJECT_VERSION}")
  
  list(LENGTH VERSION_PARTS _len)
  if(_len GREATER 0)
    list(GET VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
  endif()
  if(_len GREATER 1)
    list(GET VERSION_PARTS 1 PROJECT_VERSION_MINOR)
  endif()
  if(_len GREATER 2)
    list(GET VERSION_PARTS 2 PROJECT_VERSION_PATCH)
  endif()
 
endif()

#----------------------------------------------------------------


message(STATUS "Building SquareArray version ${PROJECT_VERSION}")

include(CheckLanguage)
check_language(CUDA)

include(GNUInstallDirs)

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

set_target_properties(square_array PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_link_libraries(main_square_array PRIVATE square_array)


#------------------------------------------------------------------------------------


install(TARGETS square_array main_square_array
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES square_array.h DESTINATION include)


