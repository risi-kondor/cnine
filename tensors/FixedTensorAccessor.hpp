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


#ifndef _CnineFixedTensorAccessor
#define _CnineFixedTensorAccessor

#include "TensorView.hpp"

namespace cnine{


  template<typename TYPE>
  class FixedTensorAccessor0{
  public:

    TYPE* arr;

    FixedTensorAccessor0(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==0 || x.ndims()==1); 
      if(x.ndims()==1) CNINE_ASSRT(x.dims[0]==1); // backward compatibility
      arr=x.get_arr();
    }

    operator TYPE(){
      return *arr;
    }

    TYPE operator()(){
      return *arr;
    }

    void set(const TYPE v){
      *arr=v;
    }

  };


  template<typename TYPE, int s0>
  class FixedTensorAccessor1{
  public:

    TYPE* arr;

    FixedTensorAccessor1(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==1);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0){
      return arr[s0*i0];
    }

    void set(const int i0, const TYPE v){
      arr[s0*i0]=v;
    }

  };


  template<typename TYPE, int s0, int s1>
  class FixedTensorAccessor2{
  public:

    TYPE* arr;

    FixedTensorAccessor2(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0, const int i1){
      return arr[s0*i0+s1*i1];
    }

    void set(const int i0, const int i1, const TYPE v){
      arr[s0*i0+s1*i1]=v;
    }

  };


  template<typename TYPE, int s0, int s1, int s2>
  class FixedTensorAccessor3{
  public:

    TYPE* arr;

    FixedTensorAccessor3(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==3);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0, const int i1, const int i2){
      return arr[s0*i0+s1*i1+s2*i2];
    }

    void set(const int i0, const int i1, const int i2, const TYPE v){
      arr[s0*i0+s1*i1+s2*i2]=v;
    }

  };


  template<typename TYPE, int s0, int s1, int s2, int s3>
  class FixedTensorAccessor4{
  public:

    TYPE* arr;

    FixedTensorAccessor4(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==4);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3){
      return arr[s0*i0+s1*i1+s2*i2+s3*i3];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const TYPE v){
      arr[s0*i0+s1*i1+s2*i2+s3*i3]=v;
    }

  };


  template<typename TYPE, int s0, int s1, int s2, int s3, int s4>
  class FixedTensorAccessor5{
  public:

    TYPE* arr;

    FixedTensorAccessor5(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==5);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3, const int i4){
      return arr[s0*i0+s1*i1+s2*i2+s3*i3+s4*i4];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const TYPE v){
      arr[s0*i0+s1*i1+s2*i2+s3*i3+s4*i4]=v;
    }

  };


  template<typename TYPE, int s0, int s1, int s2, int s3, int s4, int s5>
  class FixedTensorAccessor6{
  public:

    TYPE* arr;

    FixedTensorAccessor6(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==6);
      arr=x.get_arr();
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5){
      return arr[s0*i0+s1*i1+s2*i2+s3*i3+s4*i4+s5*i5];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const TYPE v){
      arr[s0*i0+s1*i1+s2*i2+s3*i3+s4*i4+s5*i5]=v;
    }

  };

}

#endif 

