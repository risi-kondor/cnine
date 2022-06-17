import torch 
import cnine


print("\nDefining a 4x4 tensor of zeros:\n")

A=cnine.rtensor.zero([4,4])
print("A =")
print(A)

print("ndims =",A.ndims())
print("dim(0) =",A.dim(0))
dims=A.dims()
print("dims[0] =",dims[0])


print("\n\nManipulating tensor elements:\n")

A=cnine.rtensor.sequential(4,4)
print("A =")
print(A)

print("A(1,2) =",A([1,2]))
print("A(1,2) =",A[1,2])
print("A(1,2) =",A(1,2))
print("\n")

A[[1,2]]=99
print("A =")
print(A)


print("\nSimple tensor operations:\n")

B=cnine.rtensor.ones(4,4)

print("A+B =")
print(A+B)
print("A*5 =")
print(A*5)
print("A*A =")
print(A*A)

A+=B
print("A =")
print(A)
A-=B
print("A =")
print(A)


print("\nConverting from/to PyTorch tensors:\n")

A=torch.rand([3,3])
B=cnine.rtensor(A)
print("B =")
print(B)
B+=B
C=B.torch()
print("C =")
print(C)


print("\n\nFunctions of tensors:\n")

A=cnine.rtensor.randn(4,4)
B=cnine.rtensor.ones([4,4])
print("inp(A,B) =",cnine.inp(A,B),"\n")
print("norm2(A) =",cnine.norm2(A),"\n")
print("ReLU(A) =")
print(cnine.ReLU(A))


print("\nTransformations of tensors:\n")

A=cnine.rtensor.sequential(4,4)
print("A.transp() =")
print(A.transp())

print("A.slice(1,2) =")
print(A.slice(1,2))

print("A.reshape([2,8]) =")
print(A.reshape([2,8]))

print("A =")
print(A)
