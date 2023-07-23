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


#ifndef _cachedf
#define _cachedf

#include "Cnine_base.hpp"


namespace cnine{

  template<typename OBJ> 
  class cachedf{
  public:

    OBJ* obj=nullptr;
    std::function<OBJ*()> make_obj;

    ~cachedf(){
      delete obj;
    }

    cachedf():
      make_obj([](){return nullptr;}){}

    cachedf(std::function<OBJ*()> _make_obj):
      make_obj(_make_obj){}


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ& operator()(){
      if(!obj) obj=make_obj();
      CNINE_ASSRT(obj);
      return *obj;
    }

  };

}

#endif 
