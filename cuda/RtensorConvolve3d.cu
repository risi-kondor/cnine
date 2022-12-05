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

#ifndef _RtensorConvolve3d_cu
#define _RtensorConvolve3d_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Rtensor5_view.hpp"
#include "Itensor1_view.hpp"
#include "Itensor2_view.hpp"
#include "CUDAhelpers.hpp"


// 5D case 
__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int nj0, const int nj1, const int nj2, const int na){

  int i0=blockIdx.x;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int j2=0; j1<nj2; j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[(i0+j0)*xs0+(i1+j1)*xs1+(i2+j2)*xs2+a*xs3+threadIdx.y*xs4]*
	    w[threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4];

  rarr[i0*rs0+i1*rs1+i2*rs2+threadIdx.x*rs3+threadIdx.y*rs4]+=t;
}

/*
__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3,  
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int nj0, const int nj1, const int na,
  const int xn0, const int xn1, const int padding0, const int padding1){
  
  int i0=blockIdx.x;
  int i1=blockIdx.y;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
    for(int j1=0; j1<nj1; j1++)
      for(int a=0; a<na; a++)
	t+=xarr[(i0+j0-padding0)*xs0+(i1+j1-padding1)*xs1+a*xs2+threadIdx.x*xs3]*
	  w[blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3];

  rarr[i0*rs0+i1*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=t;
}
*/

/*
_global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int ni0, const int nj0, const int nj1, const int na){

  int i0=blockIdx.y/ni0;
  int i1=blockIdx.y%ni0;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int a=0; a<na; a++)
	t+=xarr[blockIdx.x*xs0+(i0+j0)*xs1+(i1+j1)*xs2+a*xs3+threadIdx.x*xs4]*
	  w[blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3];

  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
}
*/

/*
__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int ni0, const int nj0, const int nj1, const int na, 
  const int xn0, const int xn1, const int padding0, const int padding1){

  int i0=blockIdx.y/ni0;
  int i1=blockIdx.y%ni0;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1-i1+padding1); j1++)
      for(int a=0; a<na; a++)
	t+=xarr[blockIdx.x*xs0+(i0+j0-padding0)*xs1+(i1+j1-padding1)*xs2+a*xs3+threadIdx.x*xs4]*
	  w[blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3];

  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
}
*/

/*
__global__ void RtensorConvolve2d_sparse_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, int* wdir, const int rn1, const int nj1, const int na){

  int i0=blockIdx.y/rn1;
  int i1=blockIdx.y%rn1;

  int row=blockIdx.y*blockDim.z+blockIdx.z;
  int offs=wdir[2*row];
  int n=wdir[2*row+1];
  
  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(nj1*na);
    int j1=(s/na)%nj1;
    int a=s%na;
    t+=xarr[blockIdx.x*xs0+(i0+j0)*xs1+(i1+j1)*xs2+a*xs3+threadIdx.x*xs4]*warr[offs+2*i+1]
  }
  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
  
}
*/

/*
__global__ void RtensorConvolve2d_sparse_padded_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, int* wdir, const int rn1, const int nj1, const int na,
  const int xn1, const int xn2, const int padding0, const int padding1){

  int i0=blockIdx.y/rn1;
  int i1=blockIdx.y%rn1;

  int row=blockIdx.y*blockDim.z+blockIdx.z;
  int offs=wdir[2*row];
  int n=wdir[2*row+1];
  
  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(nj1*na);
    int j1=(s/na)%nj1;
    if(i0+j0-padding0<0 || i0+j0-padding0>=xn1) continue;
    if(i1+j1-padding1<0 || i1+j1-padding1>=xn2) continue;
    int a=s%na;
    t+=xarr[blockIdx.x*xs0+(i0+j0-padding0)*xs1+(i1+j1-padding1)*xs2+a*xs3+threadIdx.x*xs4]*warr[offs+2*i+1]
  }
  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
  
}
*/

// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{


  void RtensorConvolve3d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor5_view& w, 
    const int padding0, const int padding1, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);
    CNINE_ASSRT(r.n3*r.n4<=1024);

    dim3 blocks(r.n0,r.n1,r.n2);
    dim3 threads(r.n3,r.n4);

    if(padding0==0&&padding1==0){
      RtensorConvolve2d_kernel<<<blocks,threads,0,stream>>>
	(r.arrg,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arrg,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arrg,w.s0,s.s1,w.s2,w.s3,w.s4,
	  w.n1,w.n2,w.n3,w.n4); 
    }else{
      /*
	RtensorConvolve2d_kernel<<<blocks,r.n4,0,stream>>>
	(r.arrg,r.s0,r.s1,r.s2,r.s3,r.s4,
	x.arrg,x.s0,x.s1,x.s2,x.s3,x.s4,
	w.arrg,w.s0,s.s1,w.s2,w.s3,
	r.n1,w.n1,w.n2,w.n3,
	x.n1,x.n2,padding0,padding1); 
      */
    }
  }

  /*
  void RtensorConvolve3d_cu(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w, 
    const int padding0, const int padding1, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0,r.n1*r.n2,r.n3);

    if(padding0==0&&padding1==0){
      RtensorConvolve2d_kernel<<<blocks,r.n4,0,stream>>>
	(r.arrg,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arrg,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arrg,w.s0,s.s1,w.s2,w.s3,
	  r.n1,w.n1,w.n2,w.n3); // changed x.n1 to r.n1 
    }else{
     RtensorConvolve2d_kernel<<<blocks,r.n4,0,stream>>>
	(r.arrg,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arrg,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arrg,w.s0,s.s1,w.s2,w.s3,
	  r.n1,w.n1,w.n2,w.n3,
	  x.n1,x.n2,padding0,padding1); 
    }
  }
  */

}

#endif 
