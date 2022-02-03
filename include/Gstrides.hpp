// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef __Gstrides
#define __Gstrides

#include "Cnine_base.hpp"
#include "Gdims.hpp"

namespace cnine{


  class Gstrides: public vector<int>{
  public:

    bool regular;

    Gstrides(){}

    Gstrides(const int k, const fill_raw& dummy): 
      vector<int>(k){}

    Gstrides(const Gdims& dims, const int s0=1): 
      vector<int>(dims.size()){
      int k=dims.size();
      assert(k>0);
      (*this)[k-1]=s0;
      for(int i=k-2; i>=0; i--)
	(*this)[i]=(*this)[i+1]*dims[i+1];
      regular=true;
    }


  public:

    bool is_regular() const{
      return regular;
    }

  };

}


#endif 
