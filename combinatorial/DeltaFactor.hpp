/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _DeltaFactor
#define _DeltaFactor

#include "Cnine_base.hpp"
#include <map>
#include "frational.hpp"

namespace cnine{

  template<typename KEY, typename OBJ>
  class Gbank: public unordered_map<KEY,OBJ*>{
  public:

    using unordered_map<KEY,OBJ*>::find;


    std::function<OBJ*(const KEY&)> make_obj;
    
    ~Gbank(){
      for(auto& p:*this) delete p.second;
    }

    Gbank(std::function<OBJ*(const KEY&)> _make_obj):
      make_obj(_make_obj){}

    OBJ& operator()(const KEY& key){
      auto it=find(key);
      if(it!=unordered_map<KEY,OBJ*>::end()) return *(*this)[key];
      OBJ* new_obj=make_obj(key);
      (*this)[key]=new_obj;
      return *new_obj;
    }

  };

  class DeltaSignature{
  public:
    int a,b,c;
    DeltaSignature(const int _a, const int _b, const int _c): a(_a), b(_b), c(_c){}
    bool operator==(const DeltaSignature& x) const{
      return (a==x.a)&&(b==x.b)&&(c==x.c);
    }
  };

}

namespace std{
  template<>
  struct hash<cnine::DeltaSignature>{
  public:
    size_t operator()(const cnine::DeltaSignature& x) const{
      size_t h=hash<int>()(x.a);
      h=(h<<1)^hash<int>()(x.b);
      h=(h<<1)^hash<int>()(x.c);
      return h;
    }
  };
}


namespace cnine{

  class DeltaFactor: public Gbank<DeltaSignature,frational>{
  public:

    using  Gbank<DeltaSignature,frational>::Gbank; //<DeltaSignature,frational>;
    using  Gbank<DeltaSignature,frational>::operator();

    DeltaFactor():
      Gbank<DeltaSignature,frational>([](const DeltaSignature& x){
	  const int a=x.a;
	  const int b=x.b;
	  const int c=x.c;

	  frational R=ffactorial(a+b-c);
	  return new frational(R*ffactorial(a-b+c)*ffactorial(-a+b+c)/ffactorial(a+b+c+1));
	}){}

    double operator()(const int a, const int b, const int c){
      return sqrt((*this)(DeltaSignature(a,b,c)));
    }

  };

}

#endif 
