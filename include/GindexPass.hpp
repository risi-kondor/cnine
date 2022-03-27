//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GindexPass
#define _GindexPass

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gindex.hpp"


namespace cnine{


  class GindexPass: public Gdims{
  public:

    using Gdims::Gdims;

  public:

    Gdims get_dims() const{
      return *this;
    }


    int operator()(const int i0) const{
      return i0;
    }


  public: // ---- Functional ---------------------------------------------------------------------------------


    void foreach(const function<void(const Gindex&)>& fn) const{
      int as=asize();
      for(int i=0; i<as; i++)
	fn(Gindex(i));
    }

  };


}

#endif 
