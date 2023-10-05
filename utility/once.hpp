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


#ifndef _once
#define _once

#include "Cnine_base.hpp"


namespace cnine{


  template<typename OBJ> 
  class once{
  public:

    bool done=false;
    OBJ obj;

    std::function<OBJ()> make_obj;

    ~once(){
    }

    once():
      make_obj([](){return OBJ();}){}

    once(std::function<OBJ()> _make_obj):
      make_obj(_make_obj){}


  public: // ---- Access -------------------


    OBJ& operator()(){
      if(!done){
	obj=make_obj();
	done=true;
      }
      return obj;
    }

  };




}

#endif 
