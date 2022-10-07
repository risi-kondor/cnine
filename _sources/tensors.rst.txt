*******
Tensors
*******


The core data types in cnine are real and complex multidimensional arrays. 
The corresponding classes are ``rtensor`` and ``ctensor``. 
Currently, to facilitate GP-GPU operations, only single precision arithmetic is supported. 

We describe basic tensor functionality for the ``rtensor`` class.  
The ``ctensor`` class's interface is analogous. 


================
Creating tensors
================


The following example shows how to create and print out a :math:`4\times 4` 
dimensional tensor filled with zeros.

.. code-block:: python

  >>> A=rtensor.zero([4,4])
  >>> print(A)
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]

Alternative fill patters are ``ones`` (a tensor filled with 1s), ``sequential`` (a tensor filled 
with the numbers 0,1,2,... in sequence, ``identity`` (used to construct an identity matrix), 
and ``gaussian`` (a tensor whose elements are drawn i.i.d. from a standard normal distribution). 

The number of tensor dimensions and the dimensions themselves can be read out as follows.

.. code-block:: python

  >>> A=rtensor.zero([3,4,4])
  >>> G.get_ndims() 
  3
  >>> print(A.get_dims())
  (3,4,4)
  >>> print(A.get_dim(2))
  4

=================
Tensor dimensions
=================

The number of tensor dimensions, the size of a specific dimenion and the ``gdims`` object corresponding 
to all the dimensions are accessed as follows.


.. code-block:: python

  >>> A=rtensor.zero([4,4])
  >>> A.get_ndims()
  2
  >>> A.get_dim(0)
  4
  >>> dims=A.get_dims()
  >>> dims[0]
  4

 
=========================
Accessing tensor elements
=========================



Tensor indexing in `cnine` always starts with 0 along each dimension. 
The following example shows how tensor elements can be set and read. 

.. code-block:: python

  >>> from cnine import *
  >>> A=rtensor.sequential([4,4])
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]

  >>> A([1,2])
  6.0
  >>> A[[1,2]]=99
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 99 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


=====================
Arithmetic operations
=====================

``rtensor`` supports the usual arithmetic tensor and matrix operations.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> B=rtensor.ones([4,4])
  >>> print(A+B)
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]
  >>> print(A*5)
  [ 0 5 10 15 ]
  [ 20 25 30 35 ]
  [ 40 45 50 55 ]
  [ 60 65 70 75 ]
  >>> print(A*A)
  [ 56 62 68 74 ]
  [ 152 174 196 218 ]
  [ 248 286 324 362 ]
  [ 344 398 452 506 ]

``rtensor`` also offers in-place operators.

.. code-block:: python

  >>> B=rtensor.ones([4,4])
  >>> A=rtensor.sequential([4,4])
  >>> A+=B
  >>> print(A)
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]

  >>> A-=B
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


====================
Functions of tensors
====================

The following shows how to compute the inner product 
:math:`\langle A, B\rangle=\sum_{i_1,\ldots,i_k} A_{i_1,\ldots,i_k} B_{i_1,\ldots,i_k}` 
between two tensors and the squared Frobenius norm 
:math:`\vert A\vert^2=\sum_{i_1,\ldots,i_k} \vert A_{i_1,\ldots,i_k}\vert^2`.

.. code-block:: python

  >>> A=rtensor.gaussian([4,4])
  >>> print(A)
  [ -1.23974 -0.407472 1.61201 0.399771 ]
  [ 1.3828 0.0523187 -0.904146 1.87065 ]
  [ -1.66043 -0.688081 0.0757219 1.47339 ]
  [ 0.097221 -0.89237 -0.228782 1.16493 ]
  >>> B=rtensor.ones([4,4])
  >>> inp(A,B)
  2.107801675796509
  >>> norm2(A)
  18.315340042114258


The ``ReLU`` function applies the function :math:`\textrm{ReLU}(x)=\textrm{max}(0,x)` to 
each element of the tensor.

.. code-block:: python

  >>> print(ReLU(A))
  [ 0 0 1.61201 0.399771 ]
  [ 1.3828 0.0523187 0 1.87065 ]
  [ 0 0 0.0757219 1.47339 ]
  [ 0.097221 0 0 1.16493 ]


==========
Transposes
==========

The ``transp`` method returns the transpose of a matrix.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> print(A.transp())
  [ 0 4 8 12 ]
  [ 1 5 9 13 ]
  [ 2 6 10 14 ]
  [ 3 7 11 15 ]


====================
Slices and reshaping
====================

The ``slice(i,c)`` method returns the slice of the tensor corresponding to setting the i'th index 
equal to c. ``reshape`` reinterprets the tensor as a tensor of a different shape.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> print(A.slice(1,2))
  [ 2 6 10 14 ]

  >>> A.reshape([2,8])
  >>> print(A)
  [ 0 1 2 3 4 5 6 7 ]
  [ 8 9 10 11 12 13 14 15 ]


=================
GPU functionality
=================

Tensors can moved back and forth between the host (CPU) and the GPU with the ``to`` method. 

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> B=A.to(1) # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(0) # Move B back to the host 

Almost all operations that `cnine` offers on the host are also available on the GPU. 
In general, if the operands are on the host, the operation will be performed on the host and 
the result is placed on the host. Conversely, if the operands are on the GPU, 
the operation will be performed on the GPU and the result placed on the same GPU.


================
gdims and gindex
================

In the previous examples tensors dimensions and tensor indices were specified simply as lists.  
As an alternative, tensor dimensions and indices can also be specified using the specialized 
classes `gdims` and `gindex`. 

.. code-block:: python

   >>> dims=gdims([3,3,5])
   >>> print(dims)
   (3,3,5)
   >>> print(len(dims))
   >>> print(dims[2])
   5
   >>> dims[2]=7
   >>> print(dims)
   (3,3,7)
   >>> 

===============
Complex tensors
===============


The ``ctensor`` complex valued tensor class supports all the same operations as ``rtensor``. 
In addition, it also has ``conj`` and ``herm`` methods to take the conjugate 
and conjugate transpose (Hermitian conjugate) of the tensor.

.. code-block:: python

  >>> A=ctensor.gaussian([4,4])
  >>> print(A)
  [ (-1.23974,0.584898) (-0.407472,-0.660558) (1.61201,0.534755) (0.399771,-0.607787) ]
  [ (1.3828,0.74589) (0.0523187,-1.75177) (-0.904146,-0.965146) (1.87065,-0.474282) ]
  [ (-1.66043,-0.546571) (-0.688081,-0.0384917) (0.0757219,0.194947) (1.47339,-0.485144) ]
  [ (0.097221,-0.370271) (-0.89237,-1.12408) (-0.228782,1.73664) (1.16493,0.882195) ]

  >>> print(A.conj())
  [ (-1.23974,-0.584898) (-0.407472,0.660558) (1.61201,-0.534755) (0.399771,0.607787) ]
  [ (1.3828,-0.74589) (0.0523187,1.75177) (-0.904146,0.965146) (1.87065,0.474282) ]
  [ (-1.66043,0.546571) (-0.688081,0.0384917) (0.0757219,-0.194947) (1.47339,0.485144) ]
  [ (0.097221,0.370271) (-0.89237,1.12408) (-0.228782,-1.73664) (1.16493,-0.882195) ]


=================
Storage details
=================

`cnine` is designed to be able to switch between different C++ backend classes for its core data types. 
The default backend class for real tensors is ``RtensorA`` and for complex tensors is ``CtensorA``. 
``RtensorA`` stores a tensor of dimensions :math:`d_1\times\ldots\times d_k` as a single contiguous array of 
:math:`d_1 \ldots d_k` floating point numbers in row major order. 
``CtensorA`` stores a complex tensor as a single array consisting of the 
real part of the tensor followed by the imaginary part. 
To facilitate memory access on the GPU, the offset of the imaginary part is rounded up to the nearest 
multiple of 128 bytes. 

A tensor object's header, including information about tensor dimensions, strides, etc., is always resident on 
the host. When a tensor array is moved to the GPU, only the array containing the tensor entries 
is moved to the  GPU's global memory. 
