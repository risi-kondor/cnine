//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensor2_view
#define _CnineRtensor2_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

//#include "Rtensor1_view.hpp"


namespace cnine{


  class Rtensor2_view{
  public:

    float* arr;
    int n0,n1;
    int s0,s1;
    int dev=0;

  public:

    Rtensor2_view(){}

    Rtensor2_view(float* _arr): 
      arr(_arr){}

    Rtensor2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    Rtensor2_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_dims.size()==2);
      n0=_dims[0];
      n1=_dims[1];
      s0=_strides[0];
      s1=_strides[1];
    }

    Rtensor2_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_strides.is_regular());
      assert(a.is_contiguous());
      assert(b.is_contiguous());
      assert(a.is_disjoint(b));
      assert(a.covers(_dims.size(),b));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    float operator()(const int i0, const int i1) const{
      return arr[s0*i0+s1*i1];
    }

    void set(const int i0, const int i1, float x){
      arr[s0*i0+s1*i1]=x;
    }

    void inc(const int i0, const int i1, float x){
      arr[s0*i0+s1*i1]+=x;
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add_matmul_AA(const Rtensor2_view& x, const Rtensor2_view& y){
      const int I=x.n1;
      assert(x.n0==n0);
      assert(y.n1==n1);
      assert(y.n1==I);

      for(int a=0; a<n0; a++)
	for(int b=0; b<n1; b++){
	  float t=0;
	  for(int i=0; i<I; i++)
	    t+=x(a,i)*y(i,b);
	  inc(a,b,t);
	}
    }
    

  public: // ---- Other views -------------------------------------------------------------------------------


    //Rtensor1_view slice0(const int i){
    //return Rtensor1_view(arr+i*s0,arrc+i*s0,n1,s1);
    //}

    //Rtensor1_view slice1(const int i){
    //return Rtensor1_view(arr+i*s1,arrc+i*s1,n0,s0);
    //}


    Rtensor2_view transp(){
      Rtensor2_view R(arr);
      R.n0=n1;
      R.n1=n0;
      R.s0=s1;
      R.s1=s0;
      return R;
    }

    /*
    Rtensor3_view split0(const int i, const int j){
      assert(i*j==n0);
      return Rtensor3_view(arr,i,j,n1,s0*j,s0,s1);
    }

    Rtensor3_view split1(const int i, const int j){
      assert(i*j==n1);
      return Rtensor3_view(arr,n0,i,j,s0,s1*j,s1);
    }
    */


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  R(i0,i1)=(*this)(i0,i1);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor2_view& x){
      stream<<x.str(); return stream;
    }


  };



}


#endif 
