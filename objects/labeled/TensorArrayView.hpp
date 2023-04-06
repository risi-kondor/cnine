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


#ifndef _CnineTensorArrayView
#define _CnineTensorArrayView

#include "Cnine_base.hpp"
#include "TensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class TensorArrayView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> TensorView;

    using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;

    using TensorView::device;
    using TensorView::total;



  public: // ---- Constructors ------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int getN() const{
      return total()/strides.back(2);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorArrayView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


