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


#ifndef _CnineBatchedTensorView
#define _CnineBatchedTensorView

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
  class BatchedTensorView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> TensorView;
    
    using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;
    using TensorView::dev;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedTensorView(){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    BatchedTensorView(const int _b, const Gdims& _dims, const int _dev=0): 
      TensorView(_dims.prepend(_b),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    BatchedTensorView(const int _b, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0): 
      TensorView(_dims.prepend(_b),fill,_dev){}
    

  public: // ---- Copying -----------------------------------------------------------------------------------


    BatchedTensorView* clone() const{
      return new BatchedTensorView(*this);
    }


  public: // ---- Devices ------------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int getb() const{
      return dims[0];
    }

    int ndims() const{
      return dims.size()-1;
    }

    int dim(const int i) const{
      return dims[i+1];
    }

    TensorView batch(const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return TensorView(arr+strides[0]*i,dims.chunk(1),strides.chunk(1));
    }


  public: // ---- Getters ------------------------------------------------------------------------------------


    TYPE operator()(const int b, const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(ix)];
    }

    TYPE operator()(const int b, const int i0) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0)];
    }

    TYPE operator()(const int b, const int i0, const int i1) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1)];
    }

    TYPE operator()(const int b, const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2)];
    }

    TYPE operator()(const int b, const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2,i3)];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const int b, const Gindex& ix, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,ix)]=x;
    }

    void set(const int b, const int i0, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0)]=x;
    }

    void set(const int b, const int i0, const int i1,  const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0,i1)]=x;
    }

    void set(const int b, const int i0, const int i1, const int i2, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2)]=x;
    }

    void set(const int b, const int i0, const int i1, const int i2, const int i3, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0,i1,i2,i3)]=x;
    }


  public: // ---- Incrementers -------------------------------------------------------------------------------


    void inc(const int b, const Gindex& ix, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,ix)]+=x;
    }

    void inc(const int b, const int i0, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0)]+=x;
    }

    void inc(const int b, const int i0, const int i1,  const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0,i1)]+=x;
    }

    void inc(const int b, const int i0, const int i1, const int i2, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0,i1,i2)]+=x;
    }

    void inc(const int b, const int i0, const int i1, const int i2, const int i3, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(b,i0,i1,i2,i3)]+=x;
    }


  public: // ---- Lambdas -----------------------------------------------------------------------------------


    void for_each_batch(const std::function<void(const int, const TensorView& x)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }

    void for_each(const std::function<void(const int, const Gindex&, TYPE& x)>& lambda) const{
      dims.for_each_index([&](const Gindex& ix){
	  lambda(ix[0],ix.chunk(1),ix,const_cast<MemArr<TYPE>&>(arr)[strides.offs(ix)]);});
    }


  public: // ---- Index changes ------------------------------------------------------------------------------


    /*
    BatchedTensorView<TYPE> transp(){
      return BatchedTensorView<TYPE>(arr,dims.transp(),strides.transp());
    }

    BatchedTensorView<TYPE> permute_indices(const vector<int>& p){
      return BatchedTensorView<TYPE>(arr,dims.permute(p),strides.permute(p));
    }

    BatchedTensorView<TYPE> reshape(const Gdims& _dims){
      CNINE_ASSRT(_dims.asize()==asize());
      CNINE_ASSRT(is_regular());
      return BatchedTensorView<TYPE>(arr,_dims,GstridesB(_dims));
    }

    BatchedTensorView<TYPE> slice(const int d, const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(d,i,string(__PRETTY_FUNCTION__)));
      return BatchedTensorView<TYPE>(arr+strides[d]*i,dims.remove(d),strides.remove(d);
    }

    BatchedTensorView<TYPE> slice(const Gindex& ix) const{
      const int k=ix.size();
      return BatchedTensorView<TYPE>(arr+strides.chunk(0,k)(ix),dims.chunk(k),strides.chunk(k);
      }
    */
    

  public: // ---- In-place Operations ------------------------------------------------------------------------
    

  public: // ---- Cumulative Operations ----------------------------------------------------------------------


  public: // ---- Matrix multiplication ---------------------------------------------------------------------

    /*
    void add_mvprod(const BatchedTensorView& x, const BatchedTensorView& y) const{
      reconcile_devices<BatchedTensorView<TYPE> >(*this,x,y,[](const BatchedTensorView<TYPE>& r, const BatchedTensorView<TYPE>& x, const BatchedTensorView<TYPE>& y){
	  CNINE_NDIMS_IS_1(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_1(y);
	  CNINE_ASSRT(x.dims[0]==r.dims[0]);
	  CNINE_ASSRT(x.dims[1]==y.dims[0]);

	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++){
	      TYPE t=0;
	      for(int k=0; k<x.dims[1]; k++)
		t+=x(i,k)*y(k);
	      r.inc(i,t);
	    }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	  });
	}
    */
	

    /*
    void add_mvprod_T(const BatchedTensorView& x, const BatchedTensorView& y){
      reconcile_devices<BatchedTensorView<TYPE> >(*this,x,y,[](BatchedTensorView<TYPE>& r, const BatchedTensorView<TYPE>& x, const BatchedTensorView<TYPE>& y){
	  CNINE_NDIMS_IS_1(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_1(y);
	  CNINE_ASSRT(x.dims[1]==r.dims[0]);
	  CNINE_ASSRT(y.dims[0]==y.dims[0]);

	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++){
	      TYPE t=0;
	      for(int k=0; k<x.dims[1]; k++)
		t+=x(k,i)*y(k);
	      r.inc(i,t);
	    }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	});
    }
    */


    /*
    void add_mprod(const BatchedTensorView& x, const BatchedTensorView& y) const{
      reconcile_devices<BatchedTensorView<TYPE> >(*this,x,y,[](const BatchedTensorView<TYPE>& r, const BatchedTensorView<TYPE>& x, const BatchedTensorView<TYPE>& y){
	  CNINE_NDIMS_IS_2(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_2(y);
	  CNINE_ASSRT(x.dims[0]==r.dims[0]);
	  CNINE_ASSRT(y.dims[1]==r.dims[1]);
	  CNINE_ASSRT(x.dims[1]==y.dims[0]);

	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++)
	      for(int j=0; j<r.dims[1]; j++){
		TYPE t=0;
		for(int k=0; k<x.dims[1]; k++)
		  t+=x(i,k)*y(k,j);
		r.inc(i,j,t);
	      }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	});
    }
    */


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedTensorView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"BatchedTensorView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      ostringstream oss;
      for_each_batch([&](const int b, const TensorView& x){
	  oss<<indent<<"Batch "<<b<<":"<<endl;
	  oss<<indent<<x<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedTensorView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };
  

}

#endif


    /*
    BatchedTensorView(const BatchedTensorView<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }
        
    BatchedTensorView& operator=(const BatchedTensorView& x){
      CNINE_ASSRT(dims==x.dims);
      CNINE_ASSIGN_WARNING();

      if(is_contiguous() && x.is_contiguous()){
	if(device()==0){
	  if(x.device()==0) std::copy(x.mem(),x.mem()+memsize(),mem());
	  if(x.device()==1) CUDA_SAFE(cudaMemcpy(mem(),x.mem(),memsize()*sizeof(TYPE),cudaMemcpyDeviceToHost)); 
	}
	if(device()==1){
	  if(x.device()==0) CUDA_SAFE(cudaMemcpy(mem(),x.mem(),memsize()*sizeof(float),cudaMemcpyHostToDevice));
	  if(x.device()==1) CUDA_SAFE(cudaMemcpy(mem(),x.mem(),memsize()*sizeof(float),cudaMemcpyDeviceToDevice));  
	}      
      }else{
	for_each([&](const Gindex& ix, TYPE& v) {v=x(ix);});
      }

      return *this;
    }

    BatchedTensorView* clone() const{
      auto r=new BatchedTensorView(MemArr<TYPE>(dims.total(),dev),dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }
    */
    //BatchedTensorView(const MemArr<TYPE>& _arr, const int _b, const Gdims& _dims, const int const GstridesB& _strides):
    //arr(_arr),
    //dims(_dims), 
    //strides(_strides), 
    //dev(_arr.device()){
    //}

    //BatchedTensorView(const BatchedTensorView<TYPE>& x, const int _dev):
    //BatchedTensorView<TYPE>(MemArr<TYPE>(x.dims.total(),_dev),x.dims,GstridesB(x.dims)){
    //(*this)=x;
    //}



    //int total() const{
    //return dims.total();
    //}

    //TYPE* mem() const{
    //return const_cast<TYPE*>(arr.get_arr())/*+strides.offset*/;
    //}

    //TYPE& mem(const int i) const{
    //return *(const_cast<TYPE*>(arr.get_arr())+strides.offset);
    //}

