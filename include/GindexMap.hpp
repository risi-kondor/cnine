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


#ifndef _CnineGindexMap
#define _CnineGindexMap

#include "Cnine_base.hpp"
#include "gvectr.hpp"
#include "GindexSet.hpp"

namespace cnine{

  class GindexMap: public vector<vector<int> >{
  public:

    typedef vector<vector<int> > BASE;
    using BASE::BASE;

    GindexMap(const vector<vector<int> >& x):
      BASE(x){}

  public:

    int ndims() const{
      int t=0;
      for(auto& p:*this)
	t+=p.size();
      return t;
    }

  };


}

#endif 
