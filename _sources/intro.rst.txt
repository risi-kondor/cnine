`cnine` is a lightweight C++ tensor library with a CUDA backend and optional Python interface. 
`cnine` is designed to expose GP-GPU functionality to the C++ or Python programmer.

`cnine` is written by Risi Kondor at the University of Chicago and is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_.   

This document provides documentation for cnine's Python interface. Not all features in the C++ library 
are available through this interface. The documentation of the C++ API can be found in pdf format 
in the package's ``doc`` directory.

************
Installation
************

Installing cnine requires the following:

#. C++11 or higher
#. Python
#. Pybind11 
#. PyTorch (optional)

To install cnine follow these steps:

#. Download the cnine package from `github <https://github.com/risi-kondor/cnine>`_. 
#. Edit the file ``config.txt`` as necessary. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your 
   system.
 
To use `cnine` from Python, load the corresponding module the usual way with ``import cnine``. 
In the following we assume that ``from cnine import *`` has also been called,  
obviating the need to prefix all `cnine` classes with ``cnine.``.

************
Known issues
************

GPU functionality is temporarily disabled. 