cmake_minimum_required(VERSION 3.25)
project( cnine VERSION 0.2.0 LANGUAGES CXX CUDA)
include(CheckLanguage)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(WITH_CUDA "Compiling with CUDA support" ON)
if(WITH_CUDA)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED CUDA::cudart CUDA::cublas)

  else(CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA compiler found")
    set(WITH_CUDA OFF)
  endif(CMAKE_CUDA_COMPILER)
endif(WITH_CUDA)
  
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(cnine cuda/dummy.cpp)
set(cnine_include_dirs include algorithms combinatorial containers hpc math matrices tensors utility wrappers include/cmaps tensor_views
  # legacy/scalar legacy/matrix legacy/tensor legacy/backendA legacy/backendB legacy/tensor_array legacy/cell_maps legacy/cell_ops legacy/labeled legacy/ntensor legacy/ntensor/functions legacy/tensor_array/cell_ops legacy/tensor_array/cell_maps
  # tensors/abandoned_Tensor tensors/functions
)

target_include_directories(cnine PUBLIC ${cnine_include_dirs})
target_link_libraries(cnine "${TORCH_LIBRARIES}")
target_compile_definitions(cnine PUBLIC _WITH_ATEN)
target_compile_definitions(cnine PUBLIC CNINE_RANGE_CHECKING)
target_compile_definitions(cnine PUBLIC CNINE_SIZE_CHECKING)
target_compile_definitions(cnine PUBLIC CNINE_DEVICE_CHECKING)
target_compile_definitions(cnine PUBLIC WITH_FAKE_GRAD)

if(WITH_CUDA)
  target_compile_definitions(cnine PUBLIC _WITH_CUDA)
  target_compile_definitions(cnine PUBLIC _WITH_CUBLAS)
  target_compile_definitions(cnine PUBLIC WITH_FAKE_GRAD)
endif(WITH_CUDA)

target_compile_options(cnine PUBLIC -Wno-sign-compare)
target_compile_options(cnine PUBLIC -Wno-deprecated-declarations)
target_compile_options(cnine PUBLIC -Wno-unused-variable)
target_compile_options(cnine PUBLIC -Wno-unused-but-set-variable)
target_compile_options(cnine PUBLIC -Wno-reorder)
target_compile_options(cnine PUBLIC -Wno-reorder-ctor)

if(WITH_CUDA)
  # Currently not compiling
  # add_subdirectory(cuda)
endif(WITH_CUDA)

option(BUILD_PYTHON "Build the python bindings" OFF)
if(BUILD_PYTHON)
  add_subdirectory(python)
endif(BUILD_PYTHON)
