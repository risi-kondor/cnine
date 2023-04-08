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


#ifndef _CnineSparseTensor
#define _CnineSparseTensor

#include "Cnine_base.hpp"
#include "map_of_maps.hpp"
#include "Gdims.hpp"
#include "Gindex.hpp"


namespace cnine{

  template<typename TYPE>
  class SparseTensor{
  public:

    unordered_map<Gindex,TYPE> map;
    map_of_maps<int,int,TYPE> map2; 

    Gdims dims;
    bool mom=false;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SparseTensor(const Gdims& _dims):
      dims(_dims){
      if(dims.size()==2) mom=true;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_filled(const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) return map2.is_filled(ix[0],ix[1]);
      else return (map.find(ix)!=map.end());
    }


    TYPE operator()(const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) return map2(ix[0],ix[1]);
      else{
	auto it=map.find(ix);
	if(it==map.end()) return TYPE();
	return it->second;
      }
    }

    TYPE& operator()(const Gindex& ix){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) return map2(ix[0],ix[1]);
      else return map(ix);
    }

    void set(const Gindex& ix, const TYPE& x){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) map2.set(ix[0],ix[1],x);
      else map[ix]=x;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_nonzero(const std::function<void(const Gindex&, const TYPE&)>& lambda) const{
      if(mom) map2.for_each([&](const int& i, const int& j, const TYPE& x){
	  lambda(Gindex(i,j),x);});
      else{
	for(auto& p:map)
	  lambda(p.first,p.second);
      }
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
     for_each_nonzero([&](const Gindex& ix, const TYPE& x){
	 oss<<indent<<ix<<"->"<<x<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SparseTensor& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
