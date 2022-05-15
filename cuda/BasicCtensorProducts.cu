//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _BasicCtensorProducts_cu
#define _BasicCtensorProducts_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ctensor2_view.hpp"


__global__ void BasicCproduct_4_1_3_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=threadIdx.x;
  const int i2=threadIdx.y;
  const int i3=threadIdx.z;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yi+xi*yr;
}


__global__ void BasicCproduct_4_2_2_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=blockIdx.y;
  const int i2=threadIdx.x;
  const int i3=threadIdx.y;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yi+xi*yr;
}


__global__ void BasicCproduct_4_3_1_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=blockIdx.y;
  const int i2=blockIdx.z;
  const int i3=threadIdx.y;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xr*yi+xi*yr;
}



namespace cnine{

  void BasicCproduct_4_cu(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
    const int n0, const int n1, const int n2, const int n3, 
    const int xs0, const int xs1, const int xs2, const int xs3, 
    const int ys0, const int ys1, const int ys2, const int ys3, 
    const int rs0, const int rs1,  const int rs2, const int rs3, 
    const cudaStream_t& stream){

    if(n1*n2*n3<=1024){
      dim3 blocks(n0);
      dim3 threads(n1,n2,n3);
      BasicCproduct_4_1_3_kernel<<<blocks,threads,0,stream>>>(arr,arrc,y.arr,y.arrc,r.arr,r.arrc,
	s0,s1,s2,s3, x.s0,x.s1,0,x.s2, 0,y.s0,y.s1,y.s2);
      return;
    }

    if(n2*n3<=1024){
      dim3 blocks(n0,n1);
      dim3 threads(n2,n3);
      BasicCproduct_4_2_2_kernel<<<blocks,threads,0,stream>>>(arr,arrc,y.arr,y.arrc,r.arr,r.arrc,
	s0,s1,s2,s3, x.s0,x.s1,0,x.s2, 0,y.s0,y.s1,y.s2);
      return;
    }

    if(n3<=1024){
      dim3 blocks(n0,n1,n2);
      dim3 threads(n3);
      BasicCproduct_4_3_1_kernel<<<blocks,threads,0,stream>>>(arr,arrc,y.arr,y.arrc,r.arr,r.arrc,
	s0,s1,s2,s3, x.s0,x.s1,0,x.s2, 0,y.s0,y.s1,y.s2);
      return;
    }

    CNINE_COUT("Error: tensor too large for BasicCproduct_4 kernel");

  }

}
