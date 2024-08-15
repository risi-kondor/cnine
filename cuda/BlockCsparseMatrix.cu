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

#ifndef _BlockCsparseMatrix_cu
#define _BlockCsparseMatrix_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cnine_base.hpp"
#include "BlockCsparseMatrix.hpp"


template<typename TYPE>
__global__ void BSM_times_BV_kernel(TYPE* rarr, const int rs0, const int rs1, 
  const TYPE* xarr, const int xs0, const int xs1, 
  const TYPE* yarr, const int ys0, const int ys1, 
  const int xblockn, const int xblockm,
  const int* ix, const int* offsets, const int N){

  int gi=blockIdx.x;
  const int* gather_row=ix+2*N+ix[2*gi];
  int gather_n=ix[2*gi+1]-1;
  int gather_target=gather_row[0];

  for(int a=0; a<n; a++){
    TYPE t=0;
    xrow=xarr+((offsets[blockIdx.x]+a)*xblockn+threadIdx.x)*xs0;
    ycol=yarr+gather_row[a+1]*xblockm*ys0+threadIdx.y*ys1;
    for(int i=0; i<xblockm; i++)
      t+=xrow[i*xs1]*ycol[i*ys1];
    rarr[(gather_target*xblockn+threadIdx.x)*rs0+threadIdx.y*rs1]+=t;
  }

}

namespace cnine{

  template<typename TYPE>
  BSM_times_BV_cu(const TensorView<TYPE>& r, const BlockCsparseMatrix<TYPE>& x, const TensorView<TYPE>& y, 
    const cudaStream_t& stream){
    CNINE_ASSRT(r.ndims()==2);
    CNINE_ASSRT(y.ndims()==2);

    if(x.block*y.dim(1)<1024){
      dim3 threads(x.blockn,y.dim(1));
      BSM_times_BV_kernel<<<x.offsets.rmap.size(),threads,0,stream>>>
	(r.get_arr(),r.stride(0),r.stride(1),
	  x.mx.get_arr(),x.mx.stride(0),x.mx.stride(1),
	  y.get_arr(),y.stride(0),y.stride(1),
	  x.blockn,x.blockm,
	  x.gather_mapL.on_device(1),x.row_offsets_on_device(1),x.offsets.rmap.size());
      return;
    }

  }

}

#endif 