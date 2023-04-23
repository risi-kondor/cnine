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


#ifndef _CnineTensor
#define _CnineTensor

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "BatchedTensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class Tensor: public TensorView<TYPE>{
  public:

    using TensorView<TYPE>::TensorView;
    using TensorView<TYPE>::arr;
    using TensorView<TYPE>::dims;
    using TensorView<TYPE>::strides;
    using TensorView<TYPE>::dev;

    using TensorView<TYPE>::operator=;
    using TensorView<TYPE>::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Tensor(){};

    Tensor(const Gdims& _dims, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=dummy.v;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Tensor<TYPE> zero(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> constant(const Gdims& _dims, const TYPE v, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_constant<TYPE>(v),_dev);
    }

    static Tensor<TYPE> sequential(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_sequential(),_dev);
    }

    static Tensor<TYPE> gaussian(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    Tensor(const Tensor<TYPE>& x):
      Tensor(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    Tensor(const Tensor<TYPE>& x, const nowarn_flag& dummy):
      Tensor(x.dims,x.dev){
      view()=x.view();
    }
        
    Tensor(const Tensor<TYPE>&& x):
      TensorView<TYPE>(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    Tensor& operator=(const Tensor& x){
      arr=x.arr;
      return *this;
    }
    

  public: // ---- Conversions ---------------------------------------------------------------------------------


    // Doesn't work 
    //operator BatchedTensorView<TYPE>() const{
    //return TensorView(*this);
    //}


  public: // ---- Transport -----------------------------------------------------------------------------------


    Tensor(const TensorView<TYPE>& x, const int _dev):
      Tensor(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<Tensor&>(*this)=Tensor(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    Tensor(const at::Tensor& T):
      Tensor(Gdims(T),T.type().is_cuda()){
      (*this)=T;
    }

    #endif


  public: // ---- Views -------------------------------------------------------------------------------------


    Tensor(const TensorView<TYPE>& x):
      Tensor(x.dims,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    TensorView<TYPE> view(){
      return TensorView<TYPE>(*this);
    }

    const TensorView<TYPE> view() const{
      return TensorView<TYPE>(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    Tensor operator*(const TensorView<TYPE>& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	Tensor R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	Tensor R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	Tensor R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return Tensor();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Tensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"Tensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Tensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE>
  inline Tensor<TYPE> operator+(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R(x);
    R.add(y);
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> prod(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R=Tensor<TYPE>::zero(x.dims,x.dev);
    R.add_prod(x,y);
    return R;
  }
    

}

#endif
