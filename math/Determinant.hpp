/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _CnineDeterminant
#define _CnineDeterminant

#include "TensorView.hpp"


namespace cnine{

  template<typename TYPE>
  TYPE determinant(const TensorView<TYPE>& x){
    CNINE_ASSRT(x.ndims()==2);
    CNINE_ASSRT(x.dim(0)==x.dim(1));
    CNINE_ASSRT(x.dim(0)==3);

    TYPE det=x(0,0)*(x(1,1)*x(2,2)-x(1,2)*x(2,1));
    det+=x(0,1)*(x(1,2)*x(2,0)-x(1,0)*x(2,2));
    det+=x(0,2)*(x(1,0)*x(2,1)-x(1,1)*x(2,0));
    return det;  
  }

}

#endif 
