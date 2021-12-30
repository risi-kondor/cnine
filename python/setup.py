import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time 

os.environ['CUDA_HOME']='/usr/local/cuda-11-3' #doesn't work, need explicit export 
#os.environ["CC"] = "clang"
cwd = os.getcwd()

#CUDA_HOME='/usr/local/cuda'
print(torch.cuda.is_available())

#CXXFLAGS += '-D_GLIBCXX_USE_CXX11_ABI=0'

setup(name='cnine',
ext_modules=[CUDAExtension('cnine', ['cnine_py.cpp','../include/Cnine_base.cu'],
                          include_dirs=['/usr/local/cuda/include',
                                        cwd+'/../include',
                                        cwd+'/../include/cmaps',
                                        cwd+'/../objects/scalar',
                                        cwd+'/../objects/tensor',
                                        cwd+'/../objects/tensor_array',
                                        cwd+'/../objects/tensor_array/cell_ops'],
                          #                                 library_dirs=['/usr/local/cuda/lib'],
                          #                                 runtime_libraries=['cudart'],
                          #                                 libraries=['cudart','dl'],
                          extra_compile_args = {
                                                'nvcc': [   '-D_WITH_CUDA',
                                                            '-D_WITH_CUBLAS'],
                                                'cxx': ['-std=c++14',
                                                        '-Wno-sign-compare',
                                                        '-Wno-deprecated-declarations',
                                                        '-Wno-unused-variable',
                                                        '-Wno-unused-but-set-variable',
                                                        '-Wno-reorder',
                                                        '-Wno-reorder-ctor',
                                                        '-D_WITH_ATEN',
                                                        '-D_WITH_CUDA',
                                                        '-D_WITH_CUBLAS',
                                                        '-DCNINE_COPY_WARNINGS',
                                                        '-DCNINE_ASSIGN_WARNINGS',
                                                        '-DCNINE_MOVE_WARNINGS',
                                                        '-DCNINE_MOVEASSIGN_WARNINGS',
                                                        '-DCNINE_RANGE_CHECKING',
                                                        '-DCNINE_SIZE_CHECKING'
                                                             #'-D_GLIBCXX_USE_CXX11_ABI=0'
                                                        ]},
                          depends=['setup.py','cnine_py.cpp','rtensor_py.cpp','ctensor_py.cpp',
                                   'rtensorarr_py.cpp','ctensorarr_py.cpp','cmaps_py.cpp','build/*/*']
                          )], 
cmdclass={'build_ext': BuildExtension}
)

print("Compilation finished:",time.ctime(time.time()))
