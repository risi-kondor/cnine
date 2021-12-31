import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time 

# --- User settings ------------------------------------------------------------------------------------------

compile_with_cuda=True

# ------------------------------------------------------------------------------------------------------------


cwd = os.getcwd()

_include_dirs=[#'/usr/local/cuda/include',
              cwd+'/../include',
              cwd+'/../include/cmaps',
              cwd+'/../objects/scalar',
              cwd+'/../objects/tensor',
              cwd+'/../objects/tensor_array',
              cwd+'/../objects/tensor_array/cell_ops']

_nvcc_compile_args=['-D_WITH_CUDA',
                   '-D_WITH_CUBLAS'
                   ]

_cxx_compile_args=['-std=c++14',
                  '-Wno-sign-compare',
                  '-Wno-deprecated-declarations',
                  '-Wno-unused-variable',
                  '-Wno-unused-but-set-variable',
                  '-Wno-reorder',
                  '-Wno-reorder-ctor',
                  '-D_WITH_ATEN',
                  '-DCNINE_COPY_WARNINGS',
                  '-DCNINE_ASSIGN_WARNINGS',
                  '-DCNINE_MOVE_WARNINGS',
                  '-DCNINE_MOVEASSIGN_WARNINGS',
                  '-DCNINE_RANGE_CHECKING',
                  '-DCNINE_SIZE_CHECKING',
                  '-DCNINE_DEVICE_CHECKING'
                  ]

if compile_with_cuda:
    _cxx_compile_args.extend(['-D_WITH_CUDA','-D_WITH_CUBLAS'])
    
_depends=['setup.py',
          'cnine_py.cpp',
          'rtensor_py.cpp',
          'ctensor_py.cpp',
          'rtensorarr_py.cpp',
          'ctensorarr_py.cpp',
          'cmaps_py.cpp',
          'build/*/*'
          ]


# ---- Compilation commands ----------------------------------------------------------------------------------


if compile_with_cuda:

    setup(name='cnine',
          ext_modules=[CUDAExtension('cnine', ['cnine_py.cpp','../include/Cnine_base.cu'],
                                     include_dirs=_include_dirs,
                                     extra_compile_args = {
                                         'nvcc': _nvcc_compile_args,
                                         'cxx': _cxx_compile_args},
                                     depends=_depends,
                                     )], 
          cmdclass={'build_ext': BuildExtension}
        )

else:

    setup(name='cnine',
          ext_modules=[CppExtension('cnine', ['cnine_py.cpp'],
                                     include_dirs=_include_dirs,
                                     extra_compile_args = {
                                         'cxx': _cxx_compile_args},
                                     depends=_depends,
                                     )], 
          cmdclass={'build_ext': BuildExtension}
        )
    

print("Compilation finished:",time.ctime(time.time()))


#os.environ['CUDA_HOME']='/usr/local/cuda' #doesn't work, need explicit export 
#os.environ["CC"] = "clang"
#CUDA_HOME='/usr/local/cuda'
#print(torch.cuda.is_available())
