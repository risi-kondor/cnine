//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensor1_view
#define _CnineCtensor1_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


namespace cnine{


  class Ctensor1_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1;
    int s0,s1;
    int dev=0;
    
  public:

    Ctensor1_view(){}

    Ctensor1_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor1_view(float* _arr, float* _arrc, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), s0(_s0), dev(_dev){}

    Ctensor1_view(float* _arr, const int _n0, const int _s0, const int _s1, const int _coffs=1): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), s0(_s0){}

    Ctensor1_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==1);
      n0=_dims[0];
      s0=_strides[0];
    }

    Ctensor1_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const int _coffs=1):
      arr(_arr), arrc(_arr+_coffs){
      assert(_strides.is_regular());
      assert(a.is_contiguous());
      assert(a.covers(_dims.size()));
      n0=_dims.unite(a);
      s0=_strides[a.back()];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0) const{
      int t=s0*i0;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, complex<float> x){
      int t=s0*i0;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, complex<float> x){
      int t=s0*i0;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const Ctensor1_view& x){
      assert(n0==x.n0);
      for(int i=0; i<n0; i++)
	inc(i,x(i));
    }

    template<typename TYPE>
    void add(const Ctensor1_view& x, const TYPE c){
      assert(n0==x.n0);
      for(int i=0; i<n0; i++)
	inc(i,x(i)*c);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<complex<float> > gtensor() const{
      Gtensor<complex<float> > R({n0},fill::raw);
      for(int i0=0; i0<n0; i0++)
	R(i0)=(*this)(i0);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Ctensor1_view& x){
      stream<<x.str(); return stream;
    }


  };



}


#endif 
