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


#ifndef _CnineTensorArrayView
#define _CnineTensorArrayView

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

  template<typename TYPE>
  class TensorArrayView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> TensorView;

    //using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;
    using TensorView::dev;
    
    using TensorView::device;
    using TensorView::total;

    int ak=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorArrayView(const MemArr<TYPE>& _arr, const int _ak, const Gdims& _dims, const GstridesB& _strides):
      TensorView(_arr,_dims,_strides), ak(_ak){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    TensorArrayView(const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      TensorView(_adims.cat(_dims),_dev),ak(_adims.size()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorArrayView(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      TensorView(_adims.cat(_dims),fill,_dev), ak(_adims.size()){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorArrayView* clone() const{
      auto r=new TensorArrayView(MemArr<TYPE>(dims.total(),dev),ak,dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    // probably deprecated
    //TensorArrayView(const TensorView& x, const Gdims& _adims):
    //TensorView(x.arr,_adims.cat(x.dims),GstridesB(_adims.size(),fill_zero()).cat(x.strides)), ak(_adims.size()){
    //}

    TensorArrayView(const Gdims& _adims, const TensorView& x):
      TensorView(x.arr,_adims.cat(x.dims),GstridesB(_adims.size(),fill_zero()).cat(x.strides)), ak(_adims.size()){
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int nadims() const{
      return ak;
    }

    int nddims() const{
      return dims.size()-ak;
    }

    Gdims get_adims() const{
      return dims.chunk(0,ak);
    }

    int adim(const int i) const{
      return dims(i);
    }

    Gdims get_ddims() const{
      return dims.chunk(ak);
    }

    int ddim(const int i) const{
      return dims(ak+i);
    }

    Gdims get_astrides() const{
      return strides.chunk(0,ak);
    }

    Gdims get_dstrides() const{
      return strides.chunk(ak);
    }

    int getN() const{
      return total()/strides[ak-1];
    }


    TensorView operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return TensorView(arr+strides[0]*i0,get_ddims(),get_dstrides());
    }

    TensorView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return TensorView(arr+strides[0]*i0+strides[1]*i1,get_ddims(),get_dstrides());
    }

    TensorView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return TensorView(arr+strides[0]*i0+strides[1]*i1+strides[2]*i2,get_ddims(),get_dstrides());
    }

    TensorView operator()(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==ak);
      return TensorView(arr+strides(ix),get_ddims(),get_dstrides());
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_cell(const std::function<void(const Gindex&, const TensorView& x)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix));});
    }

    void apply_as_mvprod(const TensorArrayView& x, const TensorArrayView& y, 
      const std::function<void(const TensorView&, const TensorView&, const TensorView&)>& lambda){
      CNINE_ASSRT(nadims()==1);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==1);
      CNINE_ASSRT(x.dims[0]==dims[0]);
      CNINE_ASSRT(x.dims[1]==y.dims[0]);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  lambda((*this)(i),x(i,j),y(j));
    }

    void apply_as_mmprod(const TensorArrayView& x, const TensorArrayView& y, 
      const std::function<void(const TensorView&, const TensorView&, const TensorView&)>& lambda){
      CNINE_ASSRT(nadims()==2);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==2);
      CNINE_ASSRT(dims[0]==x.dims[0]);
      CNINE_ASSRT(x.dims[1]==y.dims[0]);
      CNINE_ASSRT(dims[1]==y.dims[1]);

      int I=dims[0];
      int J=dims[1];
      int K=x.dims[1];
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  for(int k=0; k<K; k++)
	    lambda((*this)(i,j),x(j,k),y(k,j));
    }


  public: // ---- Reshapings ---------------------------------------------------------------------------------

    
    TensorView<TYPE> permute_adims(const vector<int>& p){
      return TensorView<TYPE>(arr,get_adims().permute(p).cat(get_ddims()),
	get_astrides().permute(p).get_dstrides());
    }

    TensorView aflatten() const{
      return TensorView(arr,get_ddims().prepend(getN()),get_dstrides().prepend(strides[ak-1]));
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const TensorView& x) const{
      add(TensorArrayView(x,get_adims()));
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorArrayView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      ostringstream oss;
      for_each_cell([&](const Gindex& ix, const TensorView& x){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<x.str(indent+"  ")<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


