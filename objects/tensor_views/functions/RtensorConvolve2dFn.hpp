/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineRtensorConvolve2dFn
#define _CnineRtensorConvolve2dFn

#include "Rtensor5_view.hpp"
#include "CSRmatrix.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  extern void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor4_view& w, , const cudaStream_t& stream);
  extern void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w, , const cudaStream_t& stream);
  #endif


  // (b,i0,i1,a,c)*(a',j0,j1,a) ->(b,i0+j0,i1+j1,a',c) 
  class RtensorConvolve2dFn{
  public:

    void operator()(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor4_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n1==x.n1-w.n1+1);
      CNINE_ASSRT(r.n2==x.n2-w.n2+1);
      CNINE_ASSRT(r.n3==w.n0);
      CNINE_ASSRT(r.n4==x.n4);
      CNINE_ASSRT(x.n3==w.n3);

      if(r.dev==0){
	for(int b=0; b<x.n0; b++)
	  for(int i0=0; i0<r.n1; i0++)
	    for(int i1=0; i1<r.n2; i1++){
	      Rtensor2_view R(r.arr+b*r.s0+i0*r.s1+i1*r.s2, r.n3,r.n4, r.s3,r.s4, r.dev);
	      for(int j0=0; j0<w.n1; j0++){
		Rtensor2_view W(w.arr+j0*w.s1, w.n0,w.n2*w.n3, w.s0,w.s3, w.dev);
		Rtensor2_view X(x.arr+b*x.s0+(i0+j0)*x.s1+i1*x.s2, w.n2*x.n3,x.n4, x.s3,x.s4, x.dev);
		R.add_mprod(W,X);
	      }
	    }
      }
      if(r.dev==1){
	CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,stream));
      }
    }


    void operator()(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n3==w.n);
      CNINE_ASSRT(r.n4==x.n4);
      const int A=x.n3;
      const int J0=x.n1-r.n1+1;
      const int J1=x.n2-r.n2+1;
      const int C=x.n4;

      if(r.dev==0){
	for(int b=0; b<x.n0; b++){
	  Rtensor4_view X=x.slice0(b);
	  for(int i0=0; i0<r.n1; i0++)
	    for(int i1=0; i1<r.n2; i1++){
	      Rtensor2_view R(r.arr+b*r.s0+i0*r.s1+i1*r.s2, r.n3,r.n4, r.s3,r.s4, r.dev);
	      w.for_each([&](const int aout, const svec<float>& vec){
		vec.for_each([&](const int s, const float v){
		    int j0=s/(J1*A);
		    int j1=(s/A)%J1;
		    int a=s%A;
		    for(int c=0; c<C; c++)
		      R.inc(aout,c,v*X(i0+j0,i1+j1,a,c));
		  });
		});
	    }
	}
      }
      if(r.dev==1){
	CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,stream));
      }
    }


  };

}

#endif 
