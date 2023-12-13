/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _CnineGather
#define _CnineGather

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "Ltensor.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  extern void gather_cu(const Rtensor2_view& r, const Rtensor2_view& x, const GatherMapB& g, const cudaStream_t& stream);
#endif 
  

    template<typename TYPE>
    void gather_rows(TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const GatherMapB& g){
      CNINE_ASSRT(_r.ndims()==2);
      CNINE_ASSRT(_r.dim(0)%g.out_stride==0);
      CNINE_ASSRT(_x.ndims()==2);
      CNINE_ASSRT(_x.dim(0)%g.in_stride==0);

      auto r=_r.view2();
      r.n0/=g.out_stride;
      r.n1*=g.out_stride;
      auto x=_x.view2();
      x.n0/=g.in_stride;
      x.n1*=g.in_stride;
    
      if(_r.dev==0){
	CNINE_ASSRT(g.get_dev()==0);
	int N=g.size();
	for(int i=0; i<N; i++){
	  int targt=g.target(i);
	  int M=g.size_of(i);
	  for(int j=0; j<M; j++)
	    r.slice0(targt)+=x.slice0(g(i,j));
	}
      }

      if(_r.dev==1){
	CUDA_STREAM(gather_cu(r,x,g,stream));
      }
    }


    template<typename TYPE>
    Ltensor<TYPE> gather_rows(const TensorView<TYPE>& x, const GatherMapB& g){
      CNINE_ASSRT(x.ndims()==2);
      Ltensor<TYPE> r({g.getn(),x.dims(1)},0,x.dev);
      gather_rows(r,x,g);
      return r;
    }

}

#endif 
