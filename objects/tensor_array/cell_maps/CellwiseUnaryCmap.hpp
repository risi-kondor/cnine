//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CellwiseUnaryCmap
#define _CellwiseUnaryCmap

#include "Cmaps2.hpp"


namespace cnine{

  class CellwiseUnaryCmap{
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseUnaryCmap(const OP& op, ARR& r, const ARR& x, const int add_flag=0){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  if(add_flag==0) op.apply(t,x.cell(i));
	  else op.add(t,x.cell(i));
	}
      }
      if(r.dev==1){
	op.apply(*this,r,x,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int> operator()(const int i, const int j) const{
      return thrust::make_tuple(i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR cellwise(const ARR& x){
    ARR r(x,x.adims,fill::raw);
    CellwiseUnaryCmap(OP(),r,x);
    return r;
  }

  template<typename OP, typename ARR>
  void add_cellwise(ARR& r, const ARR& x){
    CellwiseUnaryCmap(OP(),r,x,1);
  }

}


#endif 
