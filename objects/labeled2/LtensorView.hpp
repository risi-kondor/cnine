/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineLtensorView
#define _CnineLtensorView

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "LList.hpp"


namespace cnine{

  template<typename TYPE>
  class LtensorView: public TensorView<TYPE>{
  public:


    Llists labels;


  public: // ---- Constructors ------------------------------------------------------------------------------

  };

}

#endif 
