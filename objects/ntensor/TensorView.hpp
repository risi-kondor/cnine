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
#include "Gdims.hpp"
#include "GstridesB.hpp"
#include "Gindex.hpp"
#include "MemArr.hpp"
#include "device_helpers.hpp"

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
  class TensorView{
  public:

    MemArr<TYPE> arr;
    Gdims dims;
    GstridesB strides;
    int dev;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorView(){}

    TensorView(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides):
      arr(_arr),
      dims(_dims), 
      strides(_strides), 
      dev(_arr.device()){
    }


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    TensorView(const Gdims& _dims, const int _dev=0): 
      TensorView(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    TensorView(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      TensorView(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    TensorView(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      TensorView(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    TensorView(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0):
      TensorView(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=dummy.v;
      move_to_device(_dev);
    }

    TensorView(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorView(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorView(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorView(_dims,_dev){
      int N=dims.total();
      //if constexpr(is_complex<TYPE>()){
      //normal_distribution<double> distr;
      //for(int i=0; i<N; i++) 
      //arr[i]=TYPE(distr(rndGen),distr(rndGen))*dummy.c;
      //}else{
	normal_distribution<double> distr;
	for(int i=0; i<N; i++) 
	  arr[i]=distr(rndGen)*dummy.c;
	//}
      move_to_device(_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorView(const TensorView<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }
        
    TensorView& operator=(const TensorView& x) const{
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

      return const_cast<TensorView&>(*this);
    }

    TensorView* clone() const{
      return new TensorView(*this);
      //auto r=new TensorView(MemArr<TYPE>(dims.total(),dev),dims,GstridesB(dims));
      //(*r)=*this;
      //return r;
    }


  public: // ---- Devices ------------------------------------------------------------------------------------


    TensorView(const TensorView<TYPE>& x, const int _dev):
      TensorView<TYPE>(MemArr<TYPE>(x.dims.total(),_dev),x.dims,GstridesB(x.dims)){
      (*this)=x;
    }


  private:
    void move_to_device(const int _dev) const{
      if(_dev==dev) return;
      TensorView t(*this,_dev);
      const_cast<MemArr<TYPE>&>(arr)=t.arr;
      const_cast<int&>(dev)=t.dev;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    // TODO complex<float> is baked in here
    TensorView(const at::Tensor& T):
      TensorView(Gdims(T),T.type().is_cuda()){
      operator=(T);
    }

    TensorView& operator=(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      CNINE_ASSRT(dims==Gdims(T));
      CNINE_ASSRT(dev==T.type().is_cuda());
      if(dev==0){
	//std::copy(T.data<TYPE>(),T.data<c10::TYPE>()+total(),arr.ptr());
	std::copy(T.data<c10::complex<float>>(),T.data<c10::complex<float>>()+total(),
	  reinterpret_cast<c10::complex<float>*>(arr.ptr()));
      }
      if(dev==1){
	//CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::c10::complex<float>>(),total()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::complex<float>>(),total()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
      }
      return *this;
    }

    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      int k=ndims();
      vector<int64_t> v(k); 
      for(int i=0; i<k; i++) v[i]=dims[i];
      at::Tensor R(at::zeros(v,torch::CPU(at::kComplexFloat))); 
      //std::copy(arr,arr+memsize,reinterpret_cast<float*>(R.data<c10::complex<float> >()));
      std::copy(arr.ptr(),arr.ptr()+dims.total(),R.data<c10::complex<float>>());
      return R;
    }

    #endif


  public: // ---- Access -------------------------------------------------------------------------------------


    int device() const{
      return dev;
    }
    
    int get_dev() const{
      return dev;
    }
    
    bool is_regular() const{
      return strides.is_regular(dims);
    }

    bool is_contiguous() const{
      return strides.is_contiguous(dims);
    }

    int ndims() const{
      return dims.size();
    }

    int dim(const int i) const{
      return dims[i];
    }

    int asize() const{
      return dims.asize();
    }

    int total() const{
      return dims.total();
    }

    int memsize() const{
      return strides.memsize(dims);
    }

    // replace with get_arro?
    //TYPE* get_arr(){
    //return arr.get_arr();
    //} 

    //const TYPE* get_arr() const{
    //return arr.get_arr();
    //} 

    //TYPE* get_arro(){
    //return arr.get_arr()+strides.offset;
    //} 

    //const TYPE* get_arro() const{
    //return arr.get_arr()+strides.offset;
    //} 

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


    TensorView<TYPE> transp(){
      return TensorView<TYPE>(arr,dims.transp(),strides.transp());
    }

    TensorView<TYPE> permute_indices(const vector<int>& p){
      return TensorView<TYPE>(arr,dims.permute(p),strides.permute(p));
    }

    TensorView<TYPE> reshape(const Gdims& _dims){
      CNINE_ASSRT(_dims.asize()==asize());
      CNINE_ASSRT(is_regular());
      return TensorView<TYPE>(arr,_dims,GstridesB(_dims));
    }

    TensorView<TYPE> slice(const int d, const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(d,i,string(__PRETTY_FUNCTION__)));
      return TensorView<TYPE>(arr+strides[d]*i,dims.remove(d),strides.remove(d)/*.inc_offset(strides[d]*i)*/);
    }

    TensorView<TYPE> slice(const Gindex& ix) const{
      const int k=ix.size();
      return TensorView<TYPE>(arr+strides.chunk(0,k)(ix),dims.chunk(k),strides.chunk(k)/*.set_offset(strides.chunk(0,k)(ix))*/);
    }

    TensorView<TYPE> unsqueeze(const int d) const{
      int s;
      if(d==0) s=strides.memsize(dims);
      else s=strides[d-1];
      return TensorView(arr,Gdims(dims).insert(d,1),GstridesB(strides).insert(d,s));
    }

    TensorView<TYPE> insert_dim(const int d, const int n) const{
      return TensorView(arr,Gdims(dims).insert(d,n),GstridesB(strides).insert(d,0));
    }

    TensorView<TYPE> cinflate(const int d, const int n) const{
      CNINE_ASSRT(dims[d]==1||dims[d]==n);
      if(dims[d]==n) return *this; 
      TensorView<TYPE> R(*this);
      R.dims[d]=n;
      R.strides[d]=0;
      return R;
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
	  CUDA_SAFE(cudaMemset(mem(),0,asize()*sizeof(TYPE)));
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


    void add(const TensorView& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize()==x.asize());
      if(dev==0){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();
	  TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr();
	  for(int i=0; i<asize(); i++) ptr[i]+=xptr[i];
	}else
	  for_each([&](const Gindex& ix, TYPE& v){v+=x(ix);});
      }
      if(dev==1){
	if(is_contiguous() && x.is_contiguous() && strides==x.strides){
	  const TYPE alpha=1.0;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.arr, 1, arr, 1));
	}else
	  CNINE_UNIMPL();
      }
    }

    void add(const TensorView& x, const TYPE c){
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
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.arr, 1, arr, 1));
	}else
	  CNINE_UNIMPL();
      }
    }

    void add_prod(const TensorView& x, const TensorView& y) const{
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      CNINE_DIMS_SAME(x);
      CNINE_DIMS_SAME(y);
      if(dev==0){
	if(is_contiguous() && x.is_contiguous() && y.is_contiguous() && strides==x.strides&& strides==y.strides){
	  TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();
	  TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr();
	  TYPE* yptr=const_cast<MemArr<TYPE>&>(y.arr).get_arr();
	  for(int i=0; i<asize(); i++) ptr[i]+=xptr[i]*yptr[i];
	}else
	  for_each([&](const Gindex& ix, TYPE& v){v+=x(ix)*y(ix);});
      }
      if(dev==1){
	CNINE_UNIMPL();
      }
    }


  public: // ---- Matrix multiplication ---------------------------------------------------------------------


    void add_mvprod(const TensorView& x, const TensorView& y) const{
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
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


    void add_mvprod_T(const TensorView& x, const TensorView& y){
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[](TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
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


    void add_mprod(const TensorView& x, const TensorView& y) const{
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
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


  public: // ---- Scalar valued operations ------------------------------------------------------------------


    TYPE diff2(const TensorView& x){
      CNINE_ASSRT(x.asize()==asize());
      TYPE t=0;
      if(is_contiguous() && x.is_contiguous()){
	for(int i=0; i<asize(); i++){
	  const TYPE a=x.arr[i]-arr[i];
	  t+=a*std::conj(a);
	}
      }else{
	for_each([&](const Gindex& ix, TYPE& v){
	    const TYPE a=x(ix)-(*this)(ix);
	    t+=a*std::conj(a);
	  });
      }
      return t;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorView"<<dims<<" ["<<strides<<"]"<<endl;
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

    friend ostream& operator<<(ostream& stream, const TensorView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


    /*
    TensorView(const TensorView<TYPE>& x, const nowarn_flag& dummy):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }
    */

    /*
    TensorView(TensorView<TYPE>&& x):
      arr(std::move(x.arr)),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev),
      regular(x.regular){
      CNINE_MOVE_WARNING();
    }
    */
    /*
    template<class S=TYPE, typename=typename std::enable_if<is_complex<S>::value, S>::type>
    TensorView(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorView(_dims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=TYPE(distr(rndGen),distr(rndGen))*dummy.c;
      //move_to_device(_dev);
    }
    */
    //template<class S=TYPE, typename std::enable_if<false,void>::is_complex<S> >* = nullptr>
    //template<class S=TYPE, typename=typename std::enable_if<!is_complex<S>::value, S>::type>
