/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensorArrayVirtual
#define _CnineTensorArrayVirtual

#include "Cnine_base.hpp"
#include "TensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE, typename BASE>
  class TensorArrayVirtual: public BASE{
  public:

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorArrayVirtual(){};

    // need this?
    TensorArrayVirtual(const Gdims& _dims, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    // need this?
    TensorArrayVirtual(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    // need this?
    TensorArrayVirtual(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorArrayVirtual(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    // need this?
    TensorArrayVirtual(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorArrayVirtual(_dims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      BASE(MemArr<TYPE>((_adims.cat(_dims)).total(),dummy,_dev),_adims.cat(_dims),GstridesB(_adims.cat(_dims))){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorArrayVirtual(_adims.cat(_dims),_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorArrayVirtual(_adims.cat(_dims),_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static TensorArrayVirtual zero(const Gdims& _dims, const int _dev=0){
      return TensorArrayVirtual (_dims,fill_zero(),_dev);
    }

    static TensorArrayVirtual sequential(const Gdims& _dims, const int _dev=0){
      return TensorArrayVirtual(_dims,fill_sequential(),_dev);
    }

    static TensorArrayVirtual gaussian(const Gdims& _dims, const int _dev=0){
      return TensorArrayVirtual(_dims,fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorArrayVirtual(const TensorArrayVirtual& x):
      TensorArrayVirtual(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    TensorArrayVirtual(const TensorArrayVirtual& x, const nowarn_flag& dummy):
      TensorArrayVirtual(x.dims,x.dev){
      view()=x.view();
    }
        
    TensorArrayVirtual(const TensorArrayVirtual&& x):
      BASE(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    TensorArrayVirtual& operator=(const TensorArrayVirtual& x){
      arr=x.arr;
      return *this;
    }
    

  public: // ---- Transport -----------------------------------------------------------------------------------


    TensorArrayVirtual(const BASE& x, const int _dev):
      TensorArrayVirtual(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<TensorArrayVirtual&>(*this)=TensorArrayVirtual(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    TensorArrayVirtual(const at::Tensor& T):
      TensorArrayVirtual(Gdims(x),T.type().is_cuda()){
      (*this)=T;
    }

    #endif


  public: // ---- Views -------------------------------------------------------------------------------------


    TensorArrayVirtual(const BASE& x):
      TensorArrayVirtual(x.dims,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    BASE view(){
      return BASE(*this);
    }

    const BASE view() const{
      return BASE(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    /*
    TensorArrayVirtual operator*(const BASE& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	TensorArrayVirtual R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	TensorArrayVirtual R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	TensorArrayVirtual R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return TensorArrayVirtual();
    }
    */


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayVirtual";
    }

  };

}
    
#endif 
