//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensor3_view
#define _CnineCtensor3_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"

#include "Ctensor2_view.hpp"


namespace cnine{


  class Ctensor3_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1,n2;
    int s0,s1,s2;
    int dev=0;

  public:

    Ctensor3_view(){}

    Ctensor3_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor3_view(float* _arr, float* _arrc, 
      const int _n0, const int _n1, const int _n2, const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    Ctensor3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _coffs=1, const int _dev=0): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    Ctensor3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==3);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
      //cout<<"diff="<<arrc-arr<<endl;
    }

    Ctensor3_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const GindexSet& c, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_strides.is_regular());
      assert(a.is_contiguous());
      assert(b.is_contiguous());
      assert(c.is_contiguous());
      assert(a.is_disjoint(b));
      assert(a.is_disjoint(c));
      assert(b.is_disjoint(c));
      assert(a.covers(_dims.size(),b,c));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      n2=_dims.unite(c);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
      s2=_strides[c.back()];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0, const int i1, const int i2) const{
      int t=s0*i0+s1*i1+s2*i2;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------



  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor2_view slice0(const int i){
      return Ctensor2_view(arr+i*s0,arrc+i*s0,n1,n2,s1,s2,dev);
    }

    Ctensor2_view slice1(const int i){
      return Ctensor2_view(arr+i*s1,arrc+i*s1,n0,n2,s0,s2,dev);
    }

    Ctensor2_view slice2(const int i){
      return Ctensor2_view(arr+i*s2,arrc+i*s2,n0,n1,s0,s1,dev);
    }

    Ctensor2_view fuse01() const{
      return Ctensor2_view(arr,arrc,n0*n1,n2,s1,s2,dev);
    }    

    Ctensor2_view fuse12() const{
      return Ctensor2_view(arr,arrc,n0,n1*n2,s0,s2,dev);
    }    


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<complex<float> > gtensor() const{
      Gtensor<complex<float> > R({n0,n1,n2},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++){
	    R(i0,i1,i2)=(*this)(i0,i1,i2);
	  }
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Ctensor3_view& x){
      stream<<x.str(); return stream;
    }

    

  };


  class Ctensor3_view_t2: public Ctensor3_view{
  public:

    int tilesize;
    int nt;
    int st;

    Ctensor3_view_t2(const Ctensor3_view& x, const int n):
      Ctensor3_view(x), tilesize(n), nt((x.n2-1)/n+1), st(x.s2*n){
    }

  };


}


#endif 
