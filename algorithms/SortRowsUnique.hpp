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

#ifndef _CnineFindPlantedSubgraphs2
#define _CnineFindPlantedSubgraphs2

#include <set>

#include "Cnine_base.hpp"
#include "Tensor.hpp"
#include "tensor1_view.hpp"


namespace cnine{

  template<typename TYPE>
  class SortRowsUnique{
  public:

    Tensor<TYPE> r;

    SortRowsUnique(const TensorView<TYPE> x){
      CNINE_ASSRT(x.dev==0);
      CNINE_ASSRT(x.ndims()==2);
      int N=x.dim(0);

      set<tensor1_view<TYPE> > hashmap;
      for(int i=0; i<N; i++)
	hashmap.insert(x.rowv(i));

      int i=0;
      Tensor<TYPE> R(Gdims(hashmap.size(),x.dim(1)));
      for(auto& p:hashmap)
	R.rowv(i++)=p;
      r=R;
    }

    operator Tensor<TYPE>(){
      return r;
    }

  };

}

#endif 

