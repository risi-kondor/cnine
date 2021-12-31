`cnine` is a lightweight C++ tensor library with a CUDA backend. 
The primary purpose of `cnine` is to provide fine grained, direct control over  
GP-GPU functionality to Python and C++ developers. 
This document gives a high level introduction to `cnine`'s Python front end.  

`cnine` was developed by Risi Kondor at the University of Chicago and is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_.   

Not all features in the C++ library are available through the Python interface. 
The documentation of the C++ API can be found in pdf format in the package's ``doc`` directory.

********
Features
********

#. Support for real and complex valued tensors.
#. Transparent block level parallelization on GPUs using the tensor array data structures. 
#. Ability to execute custom CUDA kernels in parallel across many cells of a tensor array, even in irregular patterns. 

************
Installation
************

`cnine` can be used with or without PyTorch. However, the installation script uses PyTorch's 
cpp-extension facility. 
Therefore, installation requires the following:

#. C++11 or higher
#. PyTorch
#. CUDA and CUBLAS for GPU functionality 

To install `cnine` follow these steps:

#. Download the `cnine` package from `github <https://github.com/risi-kondor/cnine>`_. 
#. Edit the file ``python/setup.py`` as necessary. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your 
   system.
 
To use `cnine` from Python, load the module the usual way with ``import cnine``. 
In the following we assume that the command ``from cnine import *`` has been issued,  
obviating the need to prefix all `cnine` classes and funnctions with ``cnine.``.

************
Known issues
************

GPU functionality is currently untested. 

***************
Troubleshooting
***************

#. If it becomes necessary to change the location where `setuptools` 
   places the compiled module, add a file called ``setup.cfg`` 
   with content 

   .. code-block:: none
   
    [install]
    prefix=<target directory where you want the module to be placed>

   in the ``python`` directory. Make sure that the new target directory is in Python's load path.

#. PyTorch requires C++ extensions to be compiled against the same version of CUDA that PyTorch 
   itself was compiled with. If this becomes an issue, it might be necessary to install an 
   alternative version of CUDA on your system and force `setuptools` to use that version by setting 
   the ``CUDA_HOME`` enironment variable, as, e.g. 

   .. code-block:: none
   
    export CUDA_HOME=/usr/local/<desired CUDA version>

