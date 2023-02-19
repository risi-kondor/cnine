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


#ifndef __Larray
#define __Larray

#include "Cnine_base.hpp"
#include "Ldims.hpp"

namespace cnine{


  class Larray: public Ldims{
  public:

    Larray(const vector<int>& x):
      Ldims(x){
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------

    string str(const string indent="") const{
      osstream oss(indent);
      oss<<"array(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
	oss<<")";
      }
      return oss.str();
    }

  };

}

#endif 

