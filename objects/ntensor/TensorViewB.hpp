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


#ifndef _CnineTensorView
#define _CnineTensorView

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
  class TensorViewB: TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> TensorView;
    
    using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;
    using TensorView::dev;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorViewB(){}

    //TensorViewB(const MemArr<TYPE>& _arr, const int _b, const Gdims& _dims, const int const GstridesB& _strides):
    //arr(_arr),
    //dims(_dims), 
    //strides(_strides), 
    //dev(_arr.device()){
    //}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    TensorViewB(const int _b, const Gdims& _dims, const int _dev=0): 
      TensorView(_dims.prepend(_b),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorViewB(const int _b, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0): 
      TensorView(_dims.prepend(_b),fill,_dev){}
    

  public: // ---- Copying -----------------------------------------------------------------------------------


    /*
    TensorViewB(const TensorViewB<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }
        
    TensorViewB& operator=(const TensorViewB& x){
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

    TensorViewB* clone() const{
      auto r=new TensorViewB(MemArr<TYPE>(dims.total(),dev),dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }
    */


  public: // ---- Devices ------------------------------------------------------------------------------------


    //TensorViewB(const TensorViewB<TYPE>& x, const int _dev):
    //TensorViewB<TYPE>(MemArr<TYPE>(x.dims.total(),_dev),x.dims,GstridesB(x.dims)){
    //(*this)=x;
    //}



  public: // ---- Access -------------------------------------------------------------------------------------


    int ndims() const{
      return dims.size()-1;
    }

    int total() const{
      return dims.total();
    }

    TYPE* mem() const{
      return const_cast<TYPE*>(arr.get_arr())/*+strides.offset*/;
    }

    //TYPE& mem(const int i) const{
    //return *(const_cast<TYPE*>(arr.get_arr())+strides.offset);
    //}


  public: // ---- Getters ------------------------------------------------------------------------------------


    TYPE operator()(const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(ix)];
    }

    TYPE operator()(const int i0) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0)];
    }

    TYPE operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1)];
    }

    TYPE operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2)];
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2,i3)];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const Gindex& ix, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(ix)]=x;
    }

    void set(const int i0, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0)]=x;
    }

    void set(const int i0, const int i1,  const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1)]=x;
    }

    void set(const int i0, const int i1, const int i2, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2)]=x;
    }

    void set(const int i0, const int i1, const int i2, const int i3, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2,i3)]=x;
    }


  public: // ---- Incrementers -------------------------------------------------------------------------------


    void inc(const Gindex& ix, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(ix)]+=x;
    }

    void inc(const int i0, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0)]+=x;
    }

    void inc(const int i0, const int i1,  const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1)]+=x;
    }

    void inc(const int i0, const int i1, const int i2, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2)]+=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const TYPE x) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2,i3)]+=x;
    }


  public: // ---- Lambdas -----------------------------------------------------------------------------------


    void for_each(const std::function<void(const Gindex&, TYPE& x)>& lambda) const{
      dims.for_each_index([&](const Gindex& ix){
	  lambda(ix,const_cast<MemArr<TYPE>&>(arr)[strides.offs(ix)]);});
    }

    //void for_each(const std::function<void(const Gindex&, TYPE x)>& lambda) const{
    //dims.for_each_index([&](const Gindex& ix){
    //  lambda(ix,arr[strides.offs(ix)]);});
    //}


  public: // ---- Index changes ------------------------------------------------------------------------------


    TensorViewB<TYPE> transp(){
      return TensorViewB<TYPE>(arr,dims.transp(),strides.transp());
    }

    TensorViewB<TYPE> permute_indices(const vector<int>& p){
      return TensorViewB<TYPE>(arr,dims.permute(p),strides.permute(p));
    }

    TensorViewB<TYPE> reshape(const Gdims& _dims){
      CNINE_ASSRT(_dims.asize()==asize());
      CNINE_ASSRT(is_regular());
      return TensorViewB<TYPE>(arr,_dims,GstridesB(_dims));
    }

    TensorViewB<TYPE> slice(const int d, const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(d,i,string(__PRETTY_FUNCTION__)));
      return TensorViewB<TYPE>(arr+strides[d]*i,dims.remove(d),strides.remove(d)/*.inc_offset(strides[d]*i)*/);
    }

    TensorViewB<TYPE> slice(const Gindex& ix) const{
      const int k=ix.size();
      return TensorViewB<TYPE>(arr+strides.chunk(0,k)(ix),dims.chunk(k),strides.chunk(k)/*.set_offset(strides.chunk(0,k)(ix))*/);
    }
    

  public: // ---- In-place Operations ------------------------------------------------------------------------


    void set_zero() const{
      if(dev==0){
	if(is_contiguous())
	  std::fill(mem(),mem()+asize(),0);
	else
	  CNINE_UNIMPL();
      }
      if(dev==1){
	if(is_contiguous()){
	  CUDA_SAFE(cudaMemset(mem(),0,asize*sizeof(TYPE)));
	}else
	  CNINE_UNIMPL();
      }
    }


    void inplace_times(const TYPE c){
      if(dev==0){
	if(is_contiguous())
	  for(int i=0; i<asize(); i++) arr[i]*=c;
	else
	  for_each([&](const Gindex& ix, const TYPE& x){x*=c;});
      }
      if(dev==1){
	if(is_contiguous()){
	  const float cr=c;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas,asize(),&cr,mem(), 1,mem(), 1));
	}else
	  CNINE_UNIMPL();
      }
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const TensorViewB& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize()==x.asize());
      if(dev==0){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr()/*+strides.offset*/;
	  TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr()/*+x.strides.offset*/;
	  for(int i=0; i<asize(); i++) ptr[i]+=xptr[i];
	}else
	  for_each([&](const Gindex& ix, TYPE& v){v+=x(ix);});
      }
      if(dev==1){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  const TYPE alpha=1.0;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	}else
	  CNINE_UNIMPL();
      }
    }

    void add(const TensorViewB& x, const TYPE c){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize()==x.asize());
      if(dev==0){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();/*+strides.offset*/
	  TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr()/*+x.strides.offset*/;
	  for(int i=0; i<asize(); i++) ptr[i]+=c*xptr[i];
	}else
	  for_each([&](const Gindex& ix, const TYPE& v){v+=c*x(ix);});
      }
      if(dev==1){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  const TYPE alpha=c;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	}else
	  CNINE_UNIMPL();
      }
    }


  public: // ---- Matrix multiplication ---------------------------------------------------------------------


    void add_mvprod(const TensorViewB& x, const TensorViewB& y) const{
      reconcile_devices<TensorViewB<TYPE> >(*this,x,y,[](const TensorViewB<TYPE>& r, const TensorViewB<TYPE>& x, const TensorViewB<TYPE>& y){
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


    void add_mvprod_T(const TensorViewB& x, const TensorViewB& y){
      reconcile_devices<TensorViewB<TYPE> >(*this,x,y,[](TensorViewB<TYPE>& r, const TensorViewB<TYPE>& x, const TensorViewB<TYPE>& y){
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


    void add_mprod(const TensorViewB& x, const TensorViewB& y) const{
      reconcile_devices<TensorViewB<TYPE> >(*this,x,y,[](const TensorViewB<TYPE>& r, const TensorViewB<TYPE>& x, const TensorViewB<TYPE>& y){
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


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorViewB";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorViewB"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      ostringstream oss;

      if(ndims()==1){
	oss<<indent<<"[ ";
	for(int i0=0; i0<dims[0]; i0++)
	  oss<<(*this)(i0)<<" ";
	oss<<"]"<<endl;
	return oss.str();
      }

      if(ndims()==2){
	for(int i0=0; i0<dims[0]; i0++){
	  oss<<indent<<"[ ";
	  for(int i1=0; i1<dims[1]; i1++)
	    oss<<(*this)(i0,i1)<<" ";
	  oss<<"]"<<endl;
	}
	return oss.str();
      }

      if(ndims()>2){
	Gdims adims=dims.chunk(0,ndims()-2);
	adims.for_each_index([&](const Gindex& ix){
	    oss<<indent<<"Slice"<<ix<<":"<<endl;
	    oss<<slice(ix).str(indent+"  ")<<endl;
	  });
      }

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorViewB<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


