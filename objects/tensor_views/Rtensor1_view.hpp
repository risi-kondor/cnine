//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensor1_view
#define _CnineRtensor1_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


namespace cnine{


  class Rtensor1_view{
  public:

    float* arr;
    int n0;
    int s0;
    int dev=0;

  public:

    Rtensor1_view(){}

    Rtensor1_view(float* _arr): 
      arr(_arr){}

    Rtensor1_view(float* _arr, const int _n0, const int _s0): 
      arr(_arr), n0(_n0), s0(_s0){}

    Rtensor1_view(float* _arr, const int _n0, const int _s0, const int _s1): 
      arr(_arr), n0(_n0), s0(_s0){}

    Rtensor1_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides):
      arr(_arr){
      assert(_dims.size()==1);
      n0=_dims[0];
      s0=_strides[0];
    }

    Rtensor1_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a):
      arr(_arr){
      assert(_strides.is_regular());
      assert(a.is_contiguous());
      assert(a.covers(_dims.size()));
      n0=_dims.unite(a);
      s0=_strides[a.back()];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    float operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      return arr[s0*i0];
    }

    void set(const int i0, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      arr[s0*i0]=x;
    }

    void inc(const int i0, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      arr[s0*i0]+=x;
    }

    Rtensor1_view block(const int i0, const int m0) const{
      return Rtensor1_view(arr+i0*s0,m0,s0,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const Rtensor1_view& y){
      assert(y.n0==n0);
      for(int i0=0; i0<n0; i0++)
	set(i0,y(i0));
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0},fill::raw);
      for(int i0=0; i0<n0; i0++)
	R(i0)=(*this)(i0);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor1_view& x){
      stream<<x.str(); return stream;
    }


  };



}


#endif 
