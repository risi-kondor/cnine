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

#ifndef _cnineGatherSlices
#define _cnineGatherSlices

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "GatherMapPack.hpp"
#include "WeightedGatherMapB.hpp"
#include "FixedkGatherMap.hpp"
#include "Ltensor.hpp"
#include "logged_timer.hpp"
#include "MultiLoop.hpp"


namespace cnine{

  class GatherSlices{
  public:

    GatherSlices(){}

    template<typename TYPE>
    void operator()(const TensorView<TYPE>& _R, const TensorView<TYPE>& _X, const GatherMapB& gmap, int in_dim, int out_dim=-1){
      if(out_dim==-1) out_dim=in_dim;
      auto p=_R.co_scrunch_except(_X,out_dim,in_dim);
      auto R(p.first);
      auto X(p.second);

      int k=R.ndims()-1;
      CNINE_ASSRT(R.dim(0)==gmap.n_out);
      CNINE_ASSRT(X.dim(0)==gmap.n_in);

      int dev=R.get_dev();
      CNINE_CPUONLY();
      CNINE_ASSRT(X.get_dev()==dev);
      
      if(dev==0){

	if(k==1){

	  int rs0=R.strides[0];
	  int rs1=R.strides[1];
	  CNINE_ASSRT(rs1==1); // get rid of this somehow?

	  int xd1=X.dims[1];
	  int xs0=X.strides[0];
	  int xs1=X.strides[1];
	  CNINE_ASSRT(xs1==1); // get rid of this somehow?

	  gmap.for_each([&](const int i, const int j){
	      stdadd(X.get_arr()+j*xs0,X.get_arr()+j*xs0+xd1*xs1,R.get_arr()+i*rs0);});

	  return;
	}
      }
    }

  };



}

#endif 
