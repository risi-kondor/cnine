//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _Combinations
#define _Combinations

#include "Cnine_base.hpp"
#include "CombinationsBank.hpp"

extern cnine::CombinationsBank combinations_bank;


namespace cnine{


  class Combinations{
  public:

    CombinationsB* obj;

    Combinations(const int n, const int m){
      obj=combinations_bank.get(n,m);
    }


  public:

    int getN() const{
      return obj->getN();
    }

    Combination operator[](const int i) const{
      return (*obj)[i];
    }

    int index(const Combination& v) const{
      return obj->index(v);
    }

    void for_each(std::function<void(Combination) > fn) const{
      return obj->for_each(fn);
    }


  public:

    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const Combination& v){
	  oss<<v.str()<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Combinations& x){
      stream<<x.str(); return stream;
    }


  };


}

#endif 
