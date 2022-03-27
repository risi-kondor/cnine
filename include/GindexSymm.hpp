//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GindexSymm
#define _GindexSymm

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gindex.hpp"


namespace cnine{

  //template<int k>
  class GindexSymm: public Gdims{
  public:

    using Gdims::Gdims;

  public:

    /*
    int getk() const{
      return k;
    }
    */

    Gdims get_dims() const{
      return *this;
    }


  public:


    int operator()(const int i0) const{
      return i0;
    }

    int operator()(const int _i0, const int _i1) const{
      int i0,i1;
      if(_i0<=_i1){i0=_i0; i1=_i1;}
      else {i0=_i1; i1=_i0;}
      return i0*(*this)[1]-i0*(i0+1)/2+i1;
    }

    int operator()(const int i0, const int i1, const int i2) const{
      return 0; 
    }

    int operator()(const Gindex& ix) const{
      if(ix.size()==1) return (*this)(ix[0]);
      if(ix.size()==2) return (*this)(ix[0],ix[1]);
      if(ix.size()==3) return (*this)(ix[0],ix[1],ix[2]);
      return (*this)(ix[0]);
      //return ix.to_int(*this);
    }


  public: // ---- Functional ---------------------------------------------------------------------------------


    void foreach(const function<void(const Gindex&)>& fn) const{
      int as=asize();
      for(int i=0; i<as; i++)
	fn(Gindex(i,*this));
    }


  };


}

#endif 
