/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2026, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensorAccessor
#define _CnineTensorAccessor

#include "TensorView.hpp"

namespace cnine{

  template<typename TYPE, int k>
  class TensorAccessor{
  };


  template<typename TYPE>
  class TensorAccessor<TYPE,1>{
  public:

    TYPE* arr;
    int s0;

    TensorAccessor(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==1);
      arr=x.get_arr();
      s0=x.stride(0);
    }

    TYPE operator()(const int i0){
      return arr[s0*i0];
    }

    void set(const int i0, const TYPE v){
      arr[s0*i0]=v;
    }

  };


  template<typename TYPE>
  class TensorAccessor<TYPE,2>{
  public:

    TYPE* arr;
    int s0;
    int s1;

    TensorAccessor(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      arr=x.get_arr();
      s0=x.stride(0);
      s1=x.stride(1);
    }

    TYPE operator()(const int i0, const int i1){
      return arr[s0*i0+s1*i1];
    }

    void set(const int i0, const int i1, const TYPE v){
      arr[s0*i0+s1*i1]=v;
    }

  };


  template<typename TYPE>
  class TensorAccessor<TYPE,3>{
  public:

    TYPE* arr;
    int s0;
    int s1;
    int s2;

    TensorAccessor(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==3);
      arr=x.get_arr();
      s0=x.stride(0);
      s1=x.stride(1);
      s2=x.stride(2);
    }

    TYPE operator()(const int i0, const int i1, const int i2){
      return arr[s0*i0+s1*i1+s2*i2];
    }

    void set(const int i0, const int i1, const int i2, const TYPE v){
      arr[s0*i0+s1*i1+s2*i2]=v;
    }

  };

}

#endif 
