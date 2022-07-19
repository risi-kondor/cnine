//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensorReduceFn
#define _CnineRtensorReduceFn

#include "RtensorView.hpp"

#ifdef _WITH_CUDA
extern void InplaceReduce1_cu(const Rtensor3_view& x, const cudaStream_t& stream);
#endif 

namespace cnine{


  class Rtensor3_view_reduce1_Fn{
  public:

    void operator()(const Rtensor3_view& x){
      
      if(x.dev==0){
	for(int i0=0; i0<x.n0; i0++){
	  for(int i2=0; i2<x.n2; i2++){
	    float t=x.arr[s0*i0+s2*i2];
	    for(int i1=1; i1<x.n1; i1++)
	      t+=x.arr[x.s0*i0+x.s1*i1+x.s2*i2];
	    x.arr[x.s0*i0+x.s2*i2]=t;
	  }
	}
      }
      
      if(x.dev==1){
	CUDA_STREAM(InplaceReduce1_cu(x,stream));
      }

    }
    
  };

}
