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


#ifndef _CnineTensorArrayViewB
#define _CnineTensorArrayViewB

#include "Cnine_base.hpp"
#include "TensorViewB.hpp"

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
  class TensorArrayViewB: public TensorArrayView<TYPE>{
  public:

    typedef TensorArrayView<TYPE> TensorArrayView;
    typedef TensorViewB<TYPE> TensorViewB;

    //using TensorView::TensorView;
    using TensorArrayView::arr;
    using TensorArrayView::dims;
    using TensorArrayView::strides;
    using TensorArrayView::dev;
    using TensorArrayView::ak;
    
    using TensorArrayView::device;
    using TensorArrayView::total;

    //int ak=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    //TensorArrayViewB(const MemArr<TYPE>& _arr, const int _ak, const Gdims& _dims, const GstridesB& _strides):
    //TensorViewB(_arr,_dims,_strides), ak(_ak){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    TensorArrayViewB(const int _b, const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      TensorArrayView(_adims.prepend(b).cat(_dims),_dev),ak(_adims.size()+1){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorArrayViewB(const int _b, const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      TensorArrayView(_adims.prepend(_b).cat(_dims),fill,_dev), ak(_adims.size()+1){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorArrayViewB* clone() const{
      auto r=new TensorArrayViewB(MemArr<TYPE>(dims.total(),dev),ak,dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    TensorArrayViewB(const Gdims& _adims, const TensorViewB& x):
      TensorViewB(x.arr,_adims.prepend(x.getb()).cat(x.dims.chunk(1)),
	GstridesB(_adims.size(),fill_zero()).cat(x.strides.chunk(1)).prepend(x.strides(0))), 
      ak(_adims.size()+1){
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
      return astrides.total();
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



    TensorViewB operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return TensorViewB(arr+astride(0)*i0,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    TensorViewB operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return TensorViewB(arr+astride(0)*i0+astride(1)*i1,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    TensorViewB operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return TensorViewB(arr+astride(0)*i0+astride(1)*i1+astride(2)*i2,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    TensorViewB operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return TensorViewB(arr+strides(ix),get_ddims().prepend(get_b()),get_dstrides().prepend(get_bstride()));
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void apply_as_mvprod(const TensorArrayViewB& x, const TensorArrayViewB& y, 
      const std::function<const TensorViewB&, const TensorViewB&, const TensorViewB&>& lambda){
      CNINE_ASSRT(nadims()==1);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==1);
      CNINE_ASSRT(x.adims(0)==adims(0));
      CNINE_ASSRT(x.adims(1)==y.adims(0));
      for(int i=0; i<adims(0); i++)
	for(int j=0; j<adims(1); j++)
	  lambda((*this)(i),x(i,j),y(j));
    }

    void apply_as_mmprod(const TensorArrayViewB& x, const TensorArrayViewB& y, 
      const std::function<const TensorViewB&, const TensorViewB&, const TensorViewB&>& lambda){
      CNINE_ASSRT(nadims()==2);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==2);
      CNINE_ASSRT(adims(0)==x.adims(0));
      CNINE_ASSRT(x.adims(1)==y.adims(0));
      CNINE_ASSRT(adims(1)==y.adims(1));

      int I=adims(0);
      int J=adims(1);
      int K=x.adims(1);
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  for(int k=0; k<K; k++)
	    lambda((*this)(i,j),x(j,k),y(k,j));
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const TensorViewB& x) const{
      add(TensorArrayViewB(get_adims(),x));
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayViewB";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorArrayViewB"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorArrayViewB<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


