//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensor2_view
#define _CnineCtensor2_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"


namespace cnine{


  class Ctensor2_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1;
    int s0,s1;

  public:

    Ctensor2_view(){}

    Ctensor2_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor2_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), s0(_s0), s1(_s1){}

    Ctensor2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _coffs=1): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), s0(_s0), s1(_s1){}

    /*
    Ctensor2_view(const Gdims& dims, const Gstrides& strides, const int _coffs=1){
      assert(dims.size()==2);
      n0=dims(0);
      n1=dims(1);
      assert(strides.size()==2);
      s0=strides[0];
      s1=strides[1];
      arrc=arr+_coffs;
    }
    */


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0, const int i1){
      int t=s0*i0+s1*i1;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, complex<float> x){
      int t=s0*i0+s1*i1;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x){
      int t=s0*i0+s1*i1;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add_matmul_AA(const Ctensor2_view& x, const Ctensor2_view& y){
      const int I=x.n1;
      assert(x.n0=n0);
      assert(y.n1=n1);
      assert(y.n1=I);

      for(int a=0; a<n0; a++)
	for(int b=0; b<n1; b++){
	  complex<float> t=0;
	  for(int i=0; i<I; i++)
	    t+=x(a,i)*y(i,b);
	  inc(a,b,t);
	}
    }
    

  public: // ---- Operations ---------------------------------------------------------------------------------


    Ctensor2_view transp(){
      Ctensor2view R(arr,arrc);
      R.n0=n1;
      R.n1=n0;
      R.s0=s1;
      R.s1=s0;
      return R;
    }



  };



}


#endif 
