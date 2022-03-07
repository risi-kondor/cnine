//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Ctensor1view_add_cu
#define _Ctensor1view_add_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

//#include "Cmaps.hpp"
#include "Ctensor1_view.hpp"
#include "AccumulateCmap.hpp"



__global__ void accumulator_kernel(float* rarr, float* xarr, const int* ptr, const float* tbl, 
  const int rstride, const int xstride, const int nlines){
  
  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);
  const int t=threadIdx.x;

  const int offs=ptr[blockIdx.x]+2;
  const int n=tbl[offs-1];
  const float* target=rarr[tbl[offs-2]*rstride];

  for(int j=0; j<nlines; j++)
    shared[j*32+t]=target[j*32+t];
  __syncthreads();
    
  for(int i=0; i<n; i++){
    float* src=xarr[tbl[offs+2*i]*xstride];
    float c=tbl[offs+2*i+1];

    for(int j=0; j<nlines; j++)
      shared[j*32+t]+=c*src[j*32+t];
    __syncthreads();
  }

  for(int j=0; j<nlines; j++)
    target[j*32+t]=shared[j*32+t];
}


namespace cnine{

  void Ctensor2view_accumulator_cu(Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask, const const cudaStream_t& stream){

    mask.prepare();

    assert(r.dev==1);
    assert(x.dev==1);
    assert(r.arrc=r.arr+1);
    assert(x.arrc=x.arr+1);
    assert(r.s1==2);
    assert(x.s1==2);
    assert(x.n1==r.n1);

    int B=map.lists.size();
    int nlines=cnine::roundup(x.n1*2,32)/32;
    assert(nlines<=384);

    accumulator_kernel<<<B,32,nlines*128,stream>>>
      (r.arr,x.arr,mask.ptrg,mask.arrg,r.s0,x.s0,nlines);
    
  }

}

#endif 
