set(cnine_cuda_source $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/Cnine_base.cu> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/RtensorUtils.cu> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/RtensorReduce.cu> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/RtensorEinsumProducts.cu>)
set_source_files_properties(${cnine_cuda_source} PROPERTIES LANGUAGE CUDA)

target_sources(cnine PUBLIC ${cnine_cuda_source})
target_compile_options(cnine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
set_target_properties(cnine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cnine PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


