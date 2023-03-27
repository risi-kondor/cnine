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

#ifndef _CnineMapOfLists
#define _CnineMapOfLists

#include "Cnine_base.hpp"
#include "IntTensor.hpp"


namespace cnine{

  template<typename KEY, typename ITEM>
  class map_of_lists{
  public:

    map<KEY,std::vector<ITEM> > data;

    void push_back(const KEY& x, const ITEM& y){
      auto it=data.find(x);
      if(it==data.end()) data[x]=vector<ITEM>({y});
      else it->second.push_back(y);
    }

    void for_each_in_list(const KEY& x, const std::function<void(const ITEM&)>& lambda){
      auto it=data.find(x);
      if(it==data.end()) return;
      auto& v=it->second;
      for(int i=0; i<v.size(); i++)
	lambda(v[i]);
    }

  };

}

#endif 
