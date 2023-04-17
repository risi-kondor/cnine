/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineBatchedTensorArrayView
#define _CnineBatchedTensorArrayView

#include "Cnine_base.hpp"
#include "TensorArrayView.hpp"
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
  class BatchedTensorArrayView: public TensorArrayView<TYPE>{
  public:

    typedef TensorArrayView<TYPE> TensorArrayView;
    typedef BatchedTensorView<TYPE> BatchedTensorView;

    using TensorArrayView::TensorArrayView;
    using TensorArrayView::arr;
    using TensorArrayView::dims;
    using TensorArrayView::strides;
    using TensorArrayView::dev;
    using TensorArrayView::ak;
    
    using TensorArrayView::device;
    using TensorArrayView::total;
    using TensorArrayView::slice;

    //int ak=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    //BatchedTensorArrayView(const MemArr<TYPE>& _arr, const int _b, const Gdims& _dims, const GstridesB& _strides):
    //BatchedTensorView(_arr,_dims,_strides), ak(_ak){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    BatchedTensorArrayView(const int _b, const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      TensorArrayView(_adims.prepend(_b),_dims,_dev){
      //ak=_adims.size()+1;
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    BatchedTensorArrayView(const int _b, const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      TensorArrayView(_adims.prepend(_b),_dims,fill,_dev){
      //ak=_adims.size()+1;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    BatchedTensorArrayView* clone() const{
      auto r=new BatchedTensorArrayView(MemArr<TYPE>(dims.total(),dev),ak,dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    BatchedTensorArrayView(const Gdims& _adims, const BatchedTensorView& x):
      TensorArrayView(x.arr,_adims.size()+1,_adims.prepend(x.getb()).cat(x.dims.chunk(1)),
	GstridesB(_adims.size(),fill_zero()).cat(x.strides.chunk(1)).prepend(x.strides(0))){
      //ak=_adims.size()+1;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getb() const{
      return dims(0);
    }

    Gdims get_bstride() const{
      return strides(0);
    }


    int nadims() const{
      return ak-1;
    }

    Gdims get_adims() const{
      return dims.chunk(1,ak-1);
    }

    int adim(const int i) const{
      return dims[i+1];
    }

    GstridesB get_astrides() const{
      return strides.chunk(1,ak-1);
    }

    int astride(const int i) const{
      return strides[i+1];
    }

    int getN() const{
      return get_adims().total();
    }


    int nddims() const{
      return dims.size()-ak;
    }

    Gdims get_ddims() const{
      return dims.chunk(ak);
    }

    int ddim(const int i) const{
      return dims[ak+i];
    }
    
    GstridesB get_dstrides() const{
      return strides.chunk(ak);
    }

    int dstride(const int i) const{
      return strides[ak+i];
    }


    TensorArrayView batch(const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return TensorArrayView(arr+strides[0]*i,nadims(),dims.chunk(1),strides.chunk(1));
    }


    BatchedTensorView operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return BatchedTensorView(arr+astride(0)*i0,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    BatchedTensorView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return BatchedTensorView(arr+astride(0)*i0+astride(1)*i1,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    BatchedTensorView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return BatchedTensorView(arr+astride(0)*i0+astride(1)*i1+astride(2)*i2,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    BatchedTensorView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return BatchedTensorView(arr+strides(ix),get_ddims().prepend(getb()),get_dstrides().prepend(get_bstride()));
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_batch(const std::function<void(const int, const TensorArrayView&)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }

    void for_each_cell(const std::function<void(const Gindex&, const BatchedTensorView&)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix));});
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void apply_as_mvprod(const BatchedTensorArrayView& x, const BatchedTensorArrayView& y, 
      const std::function<void(const BatchedTensorView&, const BatchedTensorView&, const BatchedTensorView&)>& lambda){
      CNINE_ASSRT(nadims()==1);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==1);
      CNINE_ASSRT(x.adim(0)==adim(0));
      CNINE_ASSRT(x.adim(1)==y.adim(0));
      for(int i=0; i<adim(0); i++)
	for(int j=0; j<adim(1); j++)
	  lambda((*this)(i),x(i,j),y(j));
    }

    void apply_as_mmprod(const BatchedTensorArrayView& x, const BatchedTensorArrayView& y, 
      const std::function<void(const BatchedTensorView&, const BatchedTensorView&, const BatchedTensorView&)>& lambda){
      CNINE_ASSRT(nadims()==2);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==2);
      CNINE_ASSRT(adim(0)==x.adim(0));
      CNINE_ASSRT(x.adim(1)==y.adim(0));
      CNINE_ASSRT(adim(1)==y.adim(1));

      int I=adim(0);
      int J=adim(1);
      int K=x.adim(1);
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  for(int k=0; k<K; k++)
	    lambda((*this)(i,j),x(j,k),y(k,j));
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const BatchedTensorView& x) const{
      add(BatchedTensorArrayView(get_adims(),x));
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedTensorArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"BatchedTensorArrayView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      ostringstream oss;
      if(getb()>1){
	for_each_batch([&](const int b, const TensorArrayView& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }else{
	oss<<slice(0,0).str(indent);
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedTensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


