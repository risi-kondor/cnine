*******
Tensors
*******


The core data types in cnine are real and complex multidimensional arrays. 
The corresponding classes are ``rtensor`` and ``ctensor``. 
Currently, to facilitate GP-GPU operations, `cnine` only supports single precision arithmetic. 

In the following, we describe the basic functionality of the ``rtensor`` class.  
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

Alternative fill patters are ``raw`` (storage is allocated for the tensor but its entries are not 
initialized), 
``ones`` (a tensor filled with 1s), ``sequential`` (a tensor filled 
with the numbers 0,1,2,... in sequence, ``identity`` (used to construct an identity matrix), 
and ``gaussian`` (a tensor whose elements are drawn i.i.d. from a standard normal distribution). 

For first, second and third order tensors, the list notation can be dropped. 
For example the above tensor could have been initialized simply as ``A=rtensor.zero(4,4)``. 

The number of tensor dimensions and the size of the individual dimensions can be read out as follows.

.. code-block:: python

  >>> A=rtensor.zero(4,4)
  >>> A.ndims()
  2
  >>> A.dim(0)
  4
  >>> dims=A.dims()
  >>> dims[0]
  4

 
=========================
Accessing tensor elements
=========================



Tensor indexing in `cnine` starts with 0 along each dimension. 
The following example shows how tensor elements can be set and read. 

.. code-block:: python

  >>> A=rtensor.sequential(4,4)
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]

  >>> A([1,2])
  6.0
  >>> A[1,2] # synonym for the above 
  6.0 
  >>> A(1,2) # synonym for the above, but only for tensor order up to 3
  6.0

  >>> A[[1,2]]=99
  >>> A[1,2]=99 # synonym for the above
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 99 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


..
 An alternative syntax for `reading` (but not writing) the tensor elements of  
 first, second and third order tensors indices is to pass the indices directly,  
 as in ``A(1,2)`` for ``A([1,2])``. 



=====================
Arithmetic operations
=====================

Tensors support the usual arithmetic operations of addition, subtraction, multiplication, etc..  

.. code-block:: python

  >>> A=rtensor.sequential(4,4)
  >>> B=rtensor.ones(4,4)
  >>> A+B
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]
  >>> A*5
  [ 0 5 10 15 ]
  [ 20 25 30 35 ]
  [ 40 45 50 55 ]
  [ 60 65 70 75 ]
  >>> A*A
  [ 56 62 68 74 ]
  [ 152 174 196 218 ]
  [ 248 286 324 362 ]
  [ 344 398 452 506 ]

The tensor classes also offer in-place operators.

.. code-block:: python

  >>> B=rtensor.ones(4,4)
  >>> A=rtensor.sequential(4,4)
  >>> A+=B
  >>> A
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]

  >>> A-=B
  >>> A
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


==================================
Conversion to/from Pytorch tensors
==================================

A single precision (i.e., ``float``) ``torch.Tensor`` can be converted to a cnine tensor and  
vice versa. 


.. code-block:: python

 >>> A=torch.rand([3,3])
 >>> A
 tensor([[0.8592, 0.2147, 0.0056],
	 [0.5370, 0.1644, 0.4119],
	 [0.9330, 0.2284, 0.2406]])
 >>> B=rtensor(A)
 >>> B
 [ 0.859238 0.214724 0.00555366 ]
 [ 0.536953 0.164431 0.411862 ]
 [ 0.932963 0.228432 0.240566 ]

 >>> B+=B
 >>> B
 [ 1.71848 0.429447 0.0111073 ]
 [ 1.07391 0.328861 0.823725 ]
 [ 1.86593 0.456864 0.481133 ]

 >>> C=B.torch()
 >>> C
 tensor([[1.7185, 0.4294, 0.0111],
	 [1.0739, 0.3289, 0.8237],
	 [1.8659, 0.4569, 0.4811]])


====================
Functions of tensors
====================

The following shows how to compute the inner product 
:math:`\langle A, B\rangle=\sum_{i_1,\ldots,i_k} A_{i_1,\ldots,i_k} B_{i_1,\ldots,i_k}` 
between two tensors and the squared Frobenius norm 
:math:`\vert A\vert^2=\sum_{i_1,\ldots,i_k} \vert A_{i_1,\ldots,i_k}\vert^2`.

.. code-block:: python

  >>> A=rtensor.gaussian(4,4)
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

  >>> A=rtensor.sequential(4,4)
  >>> print(A.transp())
  [ 0 4 8 12 ]
  [ 1 5 9 13 ]
  [ 2 6 10 14 ]
  [ 3 7 11 15 ]


====================
Slices and reshaping
====================

The ``slice(i,c)`` method returns the slice of the tensor where the i'th index is 
equal to c. ``reshape`` reinterprets the tensor as a tensor of a different shape.

.. code-block:: python

  >>> A=rtensor.sequential(4,4)
  >>> print(A.slice(1,2))
  [ 2 6 10 14 ]

  >>> A.reshape([2,8])
  >>> print(A)
  [ 0 1 2 3 4 5 6 7 ]
  [ 8 9 10 11 12 13 14 15 ]

By default, Python assigns objects by reference. To create an actual copy of a cnine tensor, 
use the ``copy()`` method. 


=================
GPU functionality
=================

In `cnine` device number 0 is the host (CPU) and device number 1 is the GPU. 
The optional ``device`` argument makes it possible to create a tensor on either the CPU or 
the GPU. 
 
.. code-block:: python

  >>> A=rtensor.sequential([4,4],device=1) # Create a 4x4 tensor on the GPU 

Almost all operations that `cnine` offers on the host are also available on the GPU. 
In general, if the operands are on the host, the operation will be performed on the host and 
the result is placed on the host. Conversely, if the operands are on the GPU, 
the operation will be performed on the GPU and the result will be placed on the same GPU.
The ``device`` method tells us whether a given tensor is resident on the CPU or the GPU. 

.. code-block:: python
 
 >>> A.device()
 1

Tensors can moved back and forth between the CPU and the GPU using the ``to_device`` method. 

.. code-block:: python

  >>> A=rtensor.sequential(4,4)
  >>> B=A.to(1) # Create a copy of A on the GPU
  >>> C=B.to(0) # Move B back to the host 

Support for multiple GPUs is in development. When converting a PyTorch tensor to cnine tensor 
or vice versa, the destination tensor will generally be on the same host/device as the source. 


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

  >>> A=ctensor.gaussian(4,4)
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

When converting a ``ctensor`` to a Pytorch tensor, the resulting ``torch.tensor`` acquires an 
extra dimension of size two corresponding to the real/imaginary parts of the tensor. 
In contrast to some other libraries, this extra dimension becomes the first (outer) dimension 
of the resulting tensor.

.. code-block:: python

  >>> A=ctensor.gaussian(3,3)
  >>> A
  [ (-0.121966,0.200699) (-1.08682,2.12468) (0.68429,0.371721) ]
  [ (-1.07519,1.02096) (0.0332695,0.783408) (0.744836,-0.127842) ]
  [ (0.0336061,-0.336813) (-0.526637,-0.687155) (0.462532,1.30842) ]

  >>> B=A.torch()
  >>> B
  tensor([[[-0.1220, -1.0868,  0.6843],
           [-1.0752,  0.0333,  0.7448],
           [ 0.0336, -0.5266,  0.4625]],

          [[ 0.2007,  2.1247,  0.3717],
           [ 1.0210,  0.7834, -0.1278],
           [-0.3368, -0.6872,  1.3084]]])


Pytorch tensors can be converted to ``ctensor`` objects similarly as to how they are converted to 
``rtensor`` s, as long as they follow the above format.

.. code-block:: python

  >>> A=torch.rand([3,3])
  >>> A
  tensor([[0.4290, 0.3434, 0.9867],
          [0.8319, 0.2585, 0.7662],
          [0.0404, 0.6710, 0.9135]])
  >>> A=torch.rand([2,3,3])
  >>> A
  tensor([[[0.5157, 0.5582, 0.5580],
           [0.6738, 0.0106, 0.0503],
           [0.9813, 0.5605, 0.0902]],

          [[0.2107, 0.8239, 0.4831],
           [0.2320, 0.3593, 0.6258],
           [0.3902, 0.5848, 0.7717]]])
  >>> B=ctensor(A)
  >>> B
  [ (0.515746,0.210717) (0.55815,0.823855) (0.558013,0.483094) ]
  [ (0.673824,0.231981) (0.0106288,0.35925) (0.050261,0.62581) ]
  [ (0.981344,0.39019) (0.560545,0.584823) (0.0902345,0.771727) ]



======================
Implementation details
======================

`cnine` is designed to be able to switch between different backend classes for its core data types. 
The default backend class for real tensors is ``RtensorA`` and for complex tensors is ``CtensorA``. 
``RtensorA`` stores a tensor of dimensions :math:`d_1\times\ldots\times d_k` as a single contiguous array of 
:math:`d_1 d_2 \ldots d_k` floating point numbers in row major order. 
``CtensorA`` stores a complex tensor as a single array consisting of the 
real part of the tensor followed by the imaginary part. 
To facilitate memory access on the GPU, the offset of the imaginary part is rounded up to the nearest 
multiple of 128 bytes. 

A tensor object's header, including information about tensor dimensions, strides, etc., is always stored on 
the host. When a tensor is moved to the GPU, the array containing the tensor entries 
is moved to the  GPU's global memory. 
