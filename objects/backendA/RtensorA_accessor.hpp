//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensorA_accessor
#define _CnineRtensorA_accessor

#include "PretendComplex.hpp"

namespace cnine{


  class RtensorA_accessor{
  public:

    float* arr;
    const vector<int>& strides;

  public:

    RtensorA_accessor(float* _arr, const vector<int>& _strides):
      arr(_arr), strides(_strides){
    }

  public:

    float& operator[](const int t){
      return arr[t];
    }

    float geti(const int t){
      return arr[t];
    }

    void seti(const int t, float x){
      arr[t]=x;
    }

    void inci(const int t, float x){
      arr[t]+=x;
    }

  public:

    float operator()(const int i0, const int i1){
      int t=strides[0]*i0+strides[1]*i1;
      return arr[t];
    }

    void set(const int i0, const int i1, float x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]=x;
    }

    void inc(const int i0, const int i1, float x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]+=x;
    }

  public:


  };

}

#endif 
