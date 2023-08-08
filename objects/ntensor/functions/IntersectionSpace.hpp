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

#ifndef _IntersectionSpace
#define _IntersectionSpace

#include "TensorFunctions.hpp"
#include "ComplementSpace.hpp"
//#include "SingularValueDecomposition.hpp"


namespace cnine{

  template<typename TYPE>
  class IntersectionSpace{
  public:

    Tensor<TYPE> T;
    int ncols=0;

    IntersectionSpace(const TensorView<TYPE>& X, const TensorView<TYPE>& Y){

      CNINE_ASSRT(X.ndims()==2);
      CNINE_ASSRT(Y.ndims()==2);
      CNINE_ASSRT(X.dims[1]==Y.dims[1]);

      Tensor<TYPE> B=X*Y.transp(); // a*b
      Tensor<TYPE> C=ComplementSpace(B*B.transp()-Identity<TYPE>(B.dims[0]))(); // c*a
      //cout<<"DIMS:"<<C.dims<<X.dims<<endl;
      T=C.transp()*X;
    }


    Tensor<TYPE> operator()() const{
      return T;
    }

  };

}

#endif 
