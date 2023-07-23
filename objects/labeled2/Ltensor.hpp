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


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Cnine_base.hpp"
#include "LtensorView.hpp"

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
  class Ltensor: public LtensorView<TYPE>{
  public:

    using LtensorView<TYPE>::LtensorView;
    using LtensorView<TYPE>::arr;
    using LtensorView<TYPE>::dims;
    using LtensorView<TYPE>::strides;
    using LtensorView<TYPE>::dev;

    //using LtensorView<TYPE>::operator=;
    using LtensorView<TYPE>::ndims;
    using LtensorView<TYPE>::dim;
    using LtensorView<TYPE>::set;
    using LtensorView<TYPE>::transp;


  public: // ---- Constructors ------------------------------------------------------------------------------


  public: // ---- Constructors ------------------------------------------------------------------------------


    //Tensor():
    //TensorView<TYPE>(MemArr<TYPE>(1),{1},{1}){}

    Ltensor(const Gdims& _dims, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

  };
}

#endif 
