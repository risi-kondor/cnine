//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensor4_view
#define _CnineCtensor4_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"


namespace cnine{


  class Ctensor4_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1,n2,n2;
    int s0,s1,s2,s3;

  public:

    Ctensor4_view(){}

    Ctensor4_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor4_view(float* _arr, float* _arrc, 
      const int _n0, const int _n1, const int _n2, const int _n3, 
      const int _s0, const int _s1, const int _s2, const int _s3): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), n3(_n3), s0(_s0), s1(_s1), s2(_s2), s3(_s3){}

    Ctensor4_view(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3,
      const int _s0, const int _s1, const int _s2, const int _s3, const int _coffs=1): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), n2(_n2), n3(_n3), s0(_s0), s1(_s1) s2(_s2), s3(_s3){}


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0, const int i1, const int i2, const int i3){
      int t=s0*i0+s1*i1+s2*i2+s3*i3;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2+s3*i3;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2+s3*i3;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------



  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor3_view slice0(const int i){
      return Ctensor3_view(arr+i*s0,arrc+i*s0,n1,n2,n3,s1,s2,s3);
    }

    Ctensor3_view slice1(const int i){
      return Ctensor3_view(arr+i*s1,arrc+i*s1,n0,n2,n3,s0,s2,s3);
    }

    Ctensor3_view slice2(const int i){
      return Ctensor3_view(arr+i*s2,arrc+i*s2,n0,n1,n3,s0,s1,s3);
    }

    Ctensor3_view slice3(const int i){
      return Ctensor3_view(arr+i*s3,arrc+i*s3,n0,n1,n2,s0,s1,s2);
    }


  };



}


#endif 
