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

#ifndef _CnineLtensorSpec
#define _CnineLtensorSpec

#include "LtensorSpecBase.hpp"


namespace cnine{


  template<typename TYPE>
  class Ltensor;

  template<typename TYPE>
  class LtensorSpec: public LtensorSpecBase<LtensorSpec<TYPE>>{
  public:

    typedef LtensorSpecBase<LtensorSpec<TYPE>> BASE;
    using BASE::BASE;
    LtensorSpec(){}
    LtensorSpec(const BASE& x): BASE(x){}

    Ltensor<TYPE> operator()(){
      return Ltensor<TYPE>(*this);
    }
    
  };

}

#endif 



