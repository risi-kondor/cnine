import sys,os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time
from glob import glob

def main():

 # --- User settings ------------------------------------------------------------------------------------------


 compile_with_cuda=os.environ.get("WITH_CUDA", False)

 copy_warnings=os.environ.get("COPY_WARNING", False)
 torch_convert_warnings=os.environ.get("TORCH_CONVERT_WARNINGS", False)


 # ------------------------------------------------------------------------------------------------------------


 if 'CUDA_HOME' in os.environ:
     print("CUDA found at "+os.environ['CUDA_HOME'])
 else:
     print("No CUDA found, installing without GPU support.")
     compile_with_cuda=False

 cwd = os.getcwd()

 _include_dirs=[
     cwd+'/../include',
     cwd+'/../combinatorial',
     cwd+'/../containers',
     cwd+'/../math',
     cwd+'/../utility',
     cwd+'/../wrappers',
     cwd+'/../include/cmaps',
     cwd+'/../objects/scalar',
     cwd+'/../objects/matrix',
     cwd+'/../objects/tensor',
     cwd+'/../objects/backendA',
     cwd+'/../objects/backendB',
     cwd+'/../objects/tensor_views',
     cwd+'/../objects/tensor_array',
     cwd+'/../objects/tensor_array/cell_maps',
     cwd+'/../objects/tensor_array/cell_ops',
     cwd+'/../objects/labeled',
     cwd+'/../objects/ntensor',
     cwd+'/../objects/ntensor/functions'
     ]

 _nvcc_compile_args=[
     '-D_WITH_CUDA',
     '-D_WITH_CUBLAS',
     '-DWITH_FAKE_GRAD'
     ]

 _cxx_compile_args=['-std=c++17',
                    '-Wno-sign-compare',
                    '-Wno-deprecated-declarations',
                    '-Wno-unused-variable',
                    '-Wno-unused-but-set-variable',
                    '-Wno-reorder',
                    '-Wno-reorder-ctor',
                    '-D_WITH_ATEN',
                    '-DCNINE_RANGE_CHECKING',
                    '-DCNINE_SIZE_CHECKING',
                    '-DCNINE_DEVICE_CHECKING',
                    '-DWITH_FAKE_GRAD'
                   ]

 if copy_warnings:
     _cxx_compile_args.extend([
         '-DCNINE_COPY_WARNINGS',
         '-DCNINE_ASSIGN_WARNINGS',
         '-DCNINE_MOVE_WARNINGS',
         '-DCNINE_MOVEASSIGN_WARNINGS'
         ])

 if torch_convert_warnings:
     _cxx_compile_args.extend([
         '-DCNINE_ATEN_CONVERT_WARNINGS'
         ])

 if compile_with_cuda:
     _cxx_compile_args.extend([
         '-D_WITH_CUDA',
         '-D_WITH_CUBLAS'
         ])

 _depends=['setup.py',
           'bindings/*.cpp',
           #'cnine_py.cpp',
           #'rtensor_py.cpp',
           #'ctensor_py.cpp',
           #'rtensorarr_py.cpp',
           #'ctensorarr_py.cpp',
           #'cmaps_py.cpp',
           'build/*/*'
           ]


 # ---- Compilation commands ----------------------------------------------------------------------------------


 if compile_with_cuda:
     ext_modules=[CUDAExtension('cnine_base',
                                ['bindings/cnine_py.cpp',
                                 '../include/Cnine_base.cu',
                                 '../cuda/RtensorUtils.cu',
                                 '../cuda/RtensorReduce.cu',
                                 '../cuda/RtensorEinsumProducts.cu'],
                                include_dirs=_include_dirs,
                                extra_compile_args = {
                                    'nvcc': _nvcc_compile_args,
                                    'cxx': _cxx_compile_args},
                                depends=_depends,
                                )]
 else:
     ext_modules=[CppExtension('cnine_base',
                               ['bindings/cnine_py.cpp'],
                               include_dirs=_include_dirs,
                               extra_compile_args = {
                                   'cxx': _cxx_compile_args},
                               depends=_depends,
                               )]


 setup(name='cnine',
       ext_modules=ext_modules,
       packages=find_packages('src'),
       package_dir={'': 'src'},
       py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
       include_package_data=True,
       zip_safe=False,
     cmdclass={'build_ext': BuildExtension}
 )


if __name__ == "__main__":
    main()

print("Compilation finished:",time.ctime(time.time()))


#os.environ['CUDA_HOME']='/usr/local/cuda' #doesn't work, need explicit export
#os.environ["CC"] = "clang"
#CUDA_HOME='/usr/local/cuda'
#print(torch.cuda.is_available())
