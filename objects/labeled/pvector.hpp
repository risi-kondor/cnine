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


#ifndef __pvector
#define __pvector

#include "Cnine_base.hpp"

namespace cnine{

  template<typename TYPE>
  class pvector: public vector<TYPE*>{
  public:

    using vector<TYPE*>::push_back;

    pvector(){}

    ~pvector(){
      for(auto p:*this)
	delete p;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    pvector(const pvector& x){
      for(auto p:x)
	push_back(new TYPE(*p));
    }

    pvector(pvector&& x){
      for(auto p:x)
	push_back(p);
      x.clear();
    }

    pvector& operator=(const pvector& x){
      for(auto p:*this)
	delete p;
      for(auto p:x)
	push_back(new TYPE(*p));
    }

    pvector& operator=(pvector&& x){
      for(auto p:*this)
	delete p;
      for(auto p:x)
	push_back(p);
      x.clear();
    }

  };

}

#endif 
