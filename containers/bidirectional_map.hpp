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

#ifndef _cnine_bidirectional_map
#define _cnine_bidirectional_map

#include "Cnine_base.hpp"


namespace cnine{

  template<typename KEY, typename VAL, 
	   typename Hash=std::hash<KEY>, typename Eq=std::equal_to<KEY> >
  class bidirectional_map{
  public:

    unordered_map<KEY,VAL,Hash,Eq> fmap;
    unordered_map<VAL,KEY> bmap;

  public:

    int size() const{
      return fmap.size();
    }

    pair<bool,VAL> operator()(const KEY& key) const{
      auto it=fmap.find(key);
      if(it!=fmap.end()) return make_pair(true,it->second);
      return make_pair(false,VAL());
    }

     pair<bool,KEY> backward(const VAL& key) const{
      auto it=bmap.find(key);
      if(it!=bmap.end()) return make_pair(true,it->second);
      return make_pair(false,KEY());
    }


  public:


    void insert(const KEY& key, const VAL& val){
      auto it=fmap.find(key);
      if(it!=fmap.end()){
	if(it->second==val) return;
	erase_backward(it->second);
	fmap.erase(it);
      }
      fmap.emplace(key,val);
      bmap.emplace(val,key);
      
    }


  public:


    void erase(const KEY& key){
      auto it=fmap.find(key);
      if(it!=fmap.end()){
	bmap.erase(it->second);
	fmap.erase(it);
      }
    }

    void erase_backward(const VAL& val){
      auto it=bmap.find(val);
      if(it!=bmap.end()){
	fmap.erase(it->second);
	bmap.erase(it);
      }
    }

    void clear(){
      fmap.clear();
      bmap.clear();
    }

  };

}

#endif 
