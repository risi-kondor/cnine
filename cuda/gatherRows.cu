/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _gatherRows_cu
#define _gatherRows_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"


__global__ void gatherRows_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, const int* ix, const int N, const int nc){

  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>N) return;

  int t=threadIdx.y;
  if(t>nc) return;

  int* row=ix+2*N+ix[2*i];
  int n=ix[2*i+1]-1;
  int target=row[0];

  float a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t];
  rarr[target*rs0+t]+=a;

}

__global__ void gatherRows_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, const int* g, const int N, const int K, const int nc){

  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>N) return;

  int t=threadIdx.y;
  if(t>nc) return;

  int* row=g+(K+1)*i;
  int target=row[0];

  float a=0;
  for(int j=0; j<K; j++)
    a+=xarr[row[j+1]*xs0+t];
  rarr[target*rs0+t]+=a;

}



namespace cnine{

  void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const GatherMapB& g, const cudaStream_t& stream){
    int nc=r.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(x.n1==nc);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);

    int nwarps=roundup(nc,32);
    int multi=32/nwarps;
    dim3 threads(multi,nwarps*32);

    gatherRows_kernel<<<roundup(g.size(),multi),threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,g.get_arrg(1),g.getn(),nc);
  }


  void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const FixedkGatherMap& g, const cudaStream_t& stream){
    int nc=r.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(x.n1==nc);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);
    CNINE_ASSRT(g.strides(0)==g.getk()+1);
    CNINE_ASSRT(g.strides(1)==1);

    int nwarps=roundup(nc,32);
    int multi=32/nwarps;
    dim3 threads(multi,nwarps*32);

    gatherRows_kernel<<<roundup(g.size(),multi),g.getn(),threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,g.arr.ptr(),g.size(),g.getk(),nc);
  }

}

#endif 
