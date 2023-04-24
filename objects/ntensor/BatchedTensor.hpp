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


#ifndef _CnineBatchedTensor
#define _CnineBatchedTensor

#include "Cnine_base.hpp"
#include "BatchedTensorView.hpp"
#include "Tensor.hpp"

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
  class BatchedTensor: public BatchedTensorView<TYPE>{
  public:

    typedef BatchedTensorView<TYPE> BatchedTensorView;
    typedef TensorView<TYPE> TensorView;

    using BatchedTensorView::BatchedTensorView;
    using BatchedTensorView::arr;
    using BatchedTensorView::dims;
    using BatchedTensorView::strides;
    using BatchedTensorView::dev;

    //using BatchedTensorView::operator=;
    using BatchedTensorView::for_each_batch;
    using BatchedTensorView::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedTensor(){};

    //BatchedTensor(const int _b, const Gdims& _dims, const int _dev=0):
    //VTensor(_dims.prepend(_b),_dev){}

    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //BatchedTensor(const int _b, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
    //VTensor(_dims.prepend(_b),_dev){}


  public: // ---- Lambda constructors -----------------------------------------------------------------------


    BatchedTensor(const int _b, const Gdims& _dims, const std::function<TensorView(const int)>& fn, const int _dev=0):
      BatchedTensor(_b,_dims,fill_zero(),_dev){
      for_each_batch([&](const int b, const TensorView& x){x=fn(b);});
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static BatchedTensor zero(const int _b, const Gdims& _dims, const int _dev=0){
      return BatchedTensor(_b,_dims,fill_zero(),_dev);
    }

    static BatchedTensor sequential(const int _b, const Gdims& _dims, const int _dev=0){
      return BatchedTensor(_b,_dims,fill_sequential(),_dev);
    }

    static BatchedTensor gaussian(const int _b, const Gdims& _dims, const int _dev=0){
      return BatchedTensor(_b,_dims,fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


  public: // ---- Transport -----------------------------------------------------------------------------------


  public: // ---- ATen --------------------------------------------------------------------------------------


  public: // ---- Views -------------------------------------------------------------------------------------


    BatchedTensor(const BatchedTensorView& x):
      BatchedTensor(x.getb(),x.ddims(),x.dev){
      CNINE_COPY_WARNING();
      view()=x;
    }
 
    BatchedTensorView view(){
      return BatchedTensorView(*this);
    }

    const BatchedTensorView view() const{
      return BatchedTensorView(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    /*
    BatchedTensor operator*(const BatchedTensorView& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	BatchedTensor R=zero({y.dim(1)},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	BatchedTensor R=zero({dim(0)},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	BatchedTensor R=zero({dim(0),y.dim(1)},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return BatchedTensor();
    }
    */

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedTensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"BatchedTensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedTensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE>
  inline BatchedTensor<TYPE> prod(const BatchedTensorView<TYPE>& x, const BatchedTensorView<TYPE>& y){
    BatchedTensor<TYPE> R=BatchedTensor<TYPE>::zero(std::max(x.getb(),y.getb()),x.ddims(),x.dev);
    R.add_prod(x,y);
    return R;
  }

  template<typename TYPE>
  inline BatchedTensor<TYPE> prod(const BatchedTensorView<TYPE>& x, const TensorView<TYPE>& y){
    BatchedTensor<TYPE> R=BatchedTensor<TYPE>::zero(x.getb(),x.ddims(),x.dev);
    R.add_prod(x,batch(y));
    return R;
  }

  template<typename TYPE>
  inline BatchedTensor<TYPE> prod(const TensorView<TYPE>& x, const BatchedTensorView<TYPE>& y){
    BatchedTensor<TYPE> R=BatchedTensor<TYPE>::zero(y.getb(),x.ddims(),x.dev);
    R.add_prod(batch(x),y);
    return R;
  }

}

#endif
