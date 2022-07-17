//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RtensorReduce_cu
#define _RtensorReduce_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"


__global__ void BatchedFold_kernel(float* arr, const int sb, const int s0, const int offs){
  arr[blockIdx.x*sb+threadIdx.x*s0]+=arr[blockIdx.x*sb+threadIdx.x*s0+offs];  
}


__global__ void BatchedFold_kernel1(float* arr, const int sb, const int s0, const int s1, const int offs){
  arr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1]+=arr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1+offs];  
}



namespace cnine{


  BatchedFold(float* arr, const int b, const int sb, const int n, const int s, const int offs, const cudaStream_t& stream){

    if(n<=1024){
      BatchedFold_kernel<<<b,n,0,stream>>>(arr,sb,s,offs);
      return;
    }

    dim3 blocks(b,n/1024);
    BatchedFold_kernel1<<<blocks,1024,0,stream>>>(arr,sb,s*1024,s,offs);
    if(n%1024>0) BatchedFold_kernel<<<b,n%1024,0,stream>>>(arr,sb,s,offs+(n/1024)*1024*s);

  }


  void InplaceReduce1_cu(const Rtensor3_view& x, const cudaStream_t& stream){
    int a=1; while(a<x.n1) a*=2; a/=2;

    BatchedFold(x.arr,x.n0,x.s0,(x.n1-a)*x.n2,x.s2,a*x.n2,stream);
    a/=2;

    while(a>0){
      BatchedFold(x.arr,x.n0,x.s0,a*x.n2,x.s2,a*x.n2,stream);
      a/=2;
    }

  }

}
