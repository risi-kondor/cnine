//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensor4_view
#define _CnineRtensor4_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor3_view.hpp"


namespace cnine{


  class Rtensor4_view{
  public:

    float* arr;
    int n0,n1,n2,n3;
    int s0,s1,s2,s3;
    int dev=0;

  public:

    Rtensor4_view(){}

    Rtensor4_view(float* _arr): 
      arr(_arr){}

    Rtensor4_view(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3,
      const int _s0, const int _s1, const int _s2, const int _s3, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), n2(_n2), n3(_n3), s0(_s0), s1(_s1), s2(_s2), s3(_s3), dev(_dev){}

    Rtensor4_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_dims.size()==4);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      n3=_dims[3];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
      s3=_strides[3];
    }



  public: // ---- Access ------------------------------------------------------------------------------------


    float operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3}).str()));
      return arr[s0*i0+s1*i1+s2*i2+s3*i3];
    }

    void set(const int i0, const int i1, const int i2, const int i3, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3}).str()));
      arr[s0*i0+s1*i1+s2*i2+s3*i3]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3}).str()));
      arr[s0*i0+s1*i1+s2*i2+s3*i3]+=x;
    }

    Rtensor4_view block(const int i0, const int i1, const int i2, const int i3, const int m0, const int m1, const int m2, const int m3) const{
      return Rtensor4_view(arr+i0*s0+i1*s1+i2*s2+i3*s3,m0,m1,m2,m3,s0,s1,s2,s3,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------

    
    void add(const Rtensor4_view& y){
      assert(y.n0==n0);
      assert(y.n1==n1);
      assert(y.n2==n2);
      assert(y.n3==n3);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      inc(i0,i1,i2,i3,y(i0,i1,i2,i3));
    }


  public: // ---- Other views -------------------------------------------------------------------------------


    Rtensor3_view slice0(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
	return Rtensor3_view(arr+i*s0,n1,n2,n3,s1,s2,s3,dev);
    }

    Rtensor3_view slice1(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Rtensor3_view(arr+i*s1,n0,n2,n3,s0,s2,s3,dev);
    }

    Rtensor3_view slice2(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor3_view(arr+i*s2,n0,n1,n3,s0,s1,s3,dev);
    }

    Rtensor3_view slice3(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice3(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor3_view(arr+i*s3,n0,n1,n2,s0,s1,s2,dev);
    }

    Rtensor3_view fuse01(){
      return Rtensor3_view(arr,n0*n1,n2,n3,s1,s2,s3,dev);
    }    

    Rtensor3_view fuse12(){
      return Rtensor3_view(arr,n0,n1*n2,n3,s0,s2,s3,dev);
    }    

    Rtensor3_view fuse23(){
      return Rtensor3_view(arr,n0,n1,n2*n3,s0,s1,s3,dev);
    }    


    Rtensor2_view slice01(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s1,n2,n3,s2,s3,dev);
    }

    Rtensor2_view slice02(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s2,n1,n3,s1,s3,dev);
    }

    Rtensor2_view slice03(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s3,n1,n2,s1,s2,dev);
    }

    Rtensor2_view slice12(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	    throw std::out_of_range("cnine::Rtensor4_view:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor2_view(arr+i*s1+j*s2,n0,n3,s0,s3,dev);
    }

    Rtensor2_view slice13(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s1+j*s3,n0,n2,s0,s2,dev);
    }

    Rtensor2_view slice23(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor4_view:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s2+j*s3,n0,n1,s0,s1,dev);
    }



  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1,n2,n3},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      R(i0,i1,i2,i3)=(*this)(i0,i1,i2,i3);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor4_view& x){
      stream<<x.str(); return stream;
    }


  };

  
  inline Rtensor4_view split0(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return Rtensor4_view(x.arr,i,j,x.n1,x.n2,x.s0*j,x.s0,x.s1,x.s2,x.dev);
  }

  inline Rtensor4_view split1(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n1);
    return Rtensor4_view(x.arr,x.n0,i,j,x.n2,x.s0,x.s1*j,x.s1,x.s2,x.dev);
    }

  inline Rtensor4_view split2(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n2);
    return Rtensor4_view(x.arr,x.n0,x.n1,i,j,x.s0,x.s1,x.s2*j,x.s2,x.dev);
  }
 

}


#endif 
