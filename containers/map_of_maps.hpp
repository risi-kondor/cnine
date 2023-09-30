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

#ifndef _CnineMapOfMaps
#define _CnineMapOfMaps

#include "Cnine_base.hpp"
#include <unordered_map>
#include "Tensor.hpp"


namespace cnine{

  template<typename KEY1, typename KEY2, typename TYPE>
  class map_of_maps{
  public:

    mutable unordered_map<KEY1,std::unordered_map<KEY2,TYPE> > data;

   


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      for(auto q:data)
	if(q.second->size()>0)
	  return false;
      return true;
    }

    int size() const{
      int t=0;
      for(auto& p:data)
	t+=p.second.size();
      return t;
    }

    int nfilled() const{
      int t=0;
      for(auto& p:data)
	t+=p.second.size();
      return t;
    }

    bool is_filled(const KEY1& i, const KEY2& j) const{
      auto it=data.find(i);
      if(it==data.end()) return false;
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return false;
      return true;
    }

    TYPE operator()(const KEY1& i, const KEY2& j) const{
      auto it=data.find(i);
      if(it==data.end()) return TYPE();
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return TYPE();
      return it2->second;
    }

    TYPE& operator()(const KEY1& i, const KEY2& j){
      return data[i][j];
    }

    void set(const KEY1& i, const KEY2& j, const TYPE& x){
      data[i][j]=x;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each(const std::function<void(const KEY1&, const KEY2&, const TYPE&)>& lambda) const{
      for(auto& p:data)
	for(auto& q:p.second)
	  lambda(p.first,q.first,q.second);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    
  };

}

#endif 
