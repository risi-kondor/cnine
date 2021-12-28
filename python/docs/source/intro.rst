`cnine` is a lightweight C++ tensor library with a CUDA backend, offering 
both a C++ and a Python front end API. 
 
The primary purpose of `cnine` is to provide finer grained control 
GP-GPU functionality than other tensor libraries. 

`cnine` was developed by Risi Kondor at the University of Chicago and is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_.   

This document provides documentation for cnine's Python interface. Not all features in the C++ library 
are available through this interface. The documentation of the C++ API can be found in pdf format 
in the package's ``doc`` directory.

********
Features
********

#. Support for real and complex valued tensors.
#. Transparent block level parallelization on GPUs using the tensor array data structures. 

************
Installation
************

Installing `cnine` in Python requires the following:

#. C++11 or higher
#. PyTorch

To install cnine follow these steps:

#. Download the cnine package from `github <https://github.com/risi-kondor/cnine>`_. 
#. Edit the file ``config.txt`` as necessary. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your 
   system.
 
To use `cnine` from Python, load the corresponding module the usual way with ``import cnine``. 
In the following we assume that the command ``from cnine import *`` has been issued,  
obviating the need to prefix all `cnine` classes and classes with ``cnine.``.

************
Known issues
************

GPU functionality is currently untested. 