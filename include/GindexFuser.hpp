//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GindexFuser
#define _GindexFuser

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gindex.hpp"


namespace cnine{

  //template<int k>
  class GindexFuser: public Gdims{
  public:

    //int n0,n1,n2;
    //Gdims dims;

    using Gdims::Gdims;

    //GindexFuser(const int _n0): n0(_n0){};

    //GindexFuser(const int _n0, const int _n1): n0(_n0), n1(_n1){};

    //GindexFuser(const int _n0, const int _n1, const int _n2): n0(_n0), n1(_n1), n2(_n2){};


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

    int operator()(const int i0, const int i1) const{
      return i0*(*this)[1]+i1;
    }

    int operator()(const int i0, const int i1, const int i2) const{
      return (i0*(*this)[1]+i1)*(*this)[2]+i2;
    }

    int operator()(const Gindex& ix) const{
      return ix.to_int(*this);
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
