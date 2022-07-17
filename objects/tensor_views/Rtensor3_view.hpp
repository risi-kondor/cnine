//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensor3_view
#define _CnineRtensor3_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor2_view.hpp"

#ifdef _WITH_CUDA
extern void batched_add_cu(float* rarr, const float* arr, const int b, const int sb, const int n, const int s, const cudaStream_t& stream){
#endif 


namespace cnine{


  class Rtensor3_view{
  public:

    float* arr;
    int n0,n1,n2;
    int s0,s1,s2;
    int dev=0;

  public:

    Rtensor3_view(){}

    Rtensor3_view(float* _arr): 
      arr(_arr){}

    Rtensor3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    Rtensor3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_dims.size()==3);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    float operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::Rtensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      return arr[s0*i0+s1*i1+s2*i2];
    }

    void set(const int i0, const int i1, const int i2, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::Rtensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      arr[s0*i0+s1*i1+s2*i2]=x;
    }

    void inc(const int i0, const int i1, const int i2, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::Rtensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      arr[s0*i0+s1*i1+s2*i2]+=x;
    }

    Rtensor3_view block(const int i0, const int i1, const int i2, const int m0, const int m1, const int m2) const{
      return Rtensor3_view(arr+i0*s0+i1*s1+i2*s2,m0,m1,m2,s0,s1,s2,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const Rtensor3_view& y){
      assert(y.n0==n0);
      assert(y.n1==n1);
      assert(y.n2==n2);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++){
	    inc(i0,i1,i2,y(i0,i1,i2));
	  }
    }

    void add_matmul_AA(const Rtensor3_view& x, const Rtensor2_view& y){
      const int I=x.n2;
      const int nb=x.n0;
      assert(n0==nb);
      assert(x.n1==n1);
      assert(y.n1==n1);
      assert(y.n0==I);

      for(int _b=0; _b<nb; _b++)
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    float t=0;
	    for(int i=0; i<I; i++)
	      t+=x(_b,a,i)*y(i,b);
	    inc(_b,a,b,t);
	}
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    void reduce1_destructively_into(const Rtensor2_view& r) const{
      assert(r.n0==n0);
      assert(r.n1==n2);
      reduce1_destructively();
      r.add(slice1(0));
    }


    void reduce1_destructively() const{

      if(dev==0){
	for(int i0=0; i0<n0; i0++){
	  for(int i2=0; i2<n2; i2++){
	    float t=arr[s0*i0+s2*i2];
	    for(int i1=1; i1<n1; i1++)
	      t+=arr[s0*i0+s1*i1+s2*i2];
	    arr[s0*i0+s2*i2]=t;
	  }
	}
      }
      
      if(dev==1){
	int a=1; while(a<n1) a*=2; a/=2;

	#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	batched_add_cu(x.arr,x.arr+a*x.s1,x.n0,x.s0,(x.n1-a)*x.s1,x.s2,stream);
	a/=2;
	for(;a>0;a/=2)
	  batched_add_cu(x.arr,x.arr+a*x.s1, x.n0,x.s0,a*x.s1,x.s2,stream);
	CUDA_SAFE(cudaStreamDestroy(stream));
	#endif 
      }

    }

    


  public: // ---- Other views -------------------------------------------------------------------------------


    Rtensor2_view slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor3_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      return Rtensor2_view(arr+i*s0,n1,n2,s1,s2,dev);
    }

    Rtensor2_view slice1(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor3_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      return Rtensor2_view(arr+i*s1,n0,n2,s0,s2,dev);
    }

    Rtensor2_view slice2(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor3_view:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      return Rtensor2_view(arr+i*s2,n0,n1,s0,s1,dev);
    }

    Rtensor2_view fuse01(){
      return Rtensor2_view(arr,n0*n1,n2,s1,s2,dev);
    }    

    Rtensor2_view fuse12(){
      return Rtensor2_view(arr,n0,n1*n2,s0,s2,dev);
    }    


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1,n2},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    R(i0,i1,i2)=(*this)(i0,i1,i2);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor3_view& x){
      stream<<x.str(); return stream;
    }


  };


  inline Rtensor3_view split0(const Rtensor2_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return Rtensor3_view(x.arr,i,j,x.n1,x.s0*j,x.s0,x.s1,x.dev);
  }

  inline Rtensor3_view split1(const Rtensor2_view& x, const int i, const int j){
    assert(i*j==x.n1);
    return Rtensor3_view(x.arr,x.n0,i,j,x.s0,x.s1*j,x.s1,x.dev);
  }
 

}


#endif 
