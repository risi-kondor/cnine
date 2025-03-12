set(cnine_cuda_source ${CMAKE_SOURCE_DIR}/../include/Cnine_base.cu RtensorUtils.cu RtensorReduce.cu RtensorEinsumProducts.cu)
set_source_files_properties(${cnine_cuda_source} PROPERTIES LANGUAGE CUDA)

add_library(cnine_cuda ${cnine_cuda_source})
target_link_libraries(cnine_cuda PRIVATE CUDA::cudart CUDA::cublas cnine)

set_target_properties(cnine_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cnine_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


