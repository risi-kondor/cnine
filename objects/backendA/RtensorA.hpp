//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensorA
#define _CnineRtensorA

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "RscalarA.hpp"
#include "RtensorA_accessor.hpp"

#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Rtensor4_view.hpp"
#include "RtensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  template<typename OBJ>
  class Flock;


  class RtensorArrayA;


  class RtensorA: public CnineObject,   public CnineBackendObject{
  public:

    int k;
    Gdims dims;
    //bool bundle=false;
    int nbu=-1;
    int dev=0;

    friend class RtensorArrayA;
    friend class RtensorAflock;
    friend class Flock<RtensorA>;

    //protected:

    vector<int> strides;
    int asize=0;
    int memsize=0;
    int cst=0; 

    bool is_view=false;

    float* arr=nullptr;
    float* arrg=nullptr;

  public:

    //RtensorA(){}

    ~RtensorA(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const{
      return "RtensorA";
    }

    string describe() const{
      return "RtensorA"+dims.str();
    }

    //RtensorArrayA* array_type(){
    //return reinterpret_cast<RtensorArrayA*>(this);
    //}

    //const RtensorArrayA* const_array_type(){
    //return reinterpret_cast<const RtensorArrayA*>(this);
    //}

    RtensorA():
      RtensorA(Gdims({0})){}


  // ---- Constructors -----------------------------------------------------------------------------

    
    RtensorA(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims.size()){
      CNINE_DEVICE_VALID(dev);

      k=dims.size();
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=cst; 
    }


    RtensorA(const Gdims& _dims, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims.size()){
      CNINE_DEVICE_VALID(dev)

      k=dims.size();
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=cst; 

      if(dev==0){
	arr=new float[memsize];
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }

    }


    RtensorA(const Gdims& _adims, const Gdims& _dims, const int _dev=0): // for RtensorArray
      dims(_adims,_dims), dev(_dev), strides(_adims.size()+_dims.size()){

      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("cnine error in RtensorA: device must be 0 or 1"));

      k=dims.size();
      const int ak=_adims.size();
      const int dk=_dims.size();

      strides[k-1]=1;
      for(int i=dk-2; i>=0; i--)
	strides[ak+i]=strides[ak+i+1]*_dims[i+1];
      const int cellstride=roundup(strides[ak]*_dims[0],32); 

      strides[ak-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*_adims[i+1];
      asize=strides[0]*_adims[0];

      cst=roundup(asize,32); 
      memsize=cst; 

      if(dev==0){
	arr=new float[memsize];
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }

    }


    RtensorA(const Gdims& _adims, const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): // for RtensorArray
      dims(_adims,_dims), dev(_dev), strides(_adims.size()+_dims.size()){

      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("cnine error in RtensorA: device must be 0 or 1"));

      k=dims.size();
      const int ak=_adims.size();
      const int dk=_dims.size();

      strides[k-1]=1;
      for(int i=dk-2; i>=0; i--)
	strides[ak+i]=strides[ak+i+1]*_dims[i+1];
      const int cellstride=roundup(strides[ak]*_dims[0],32); 

      strides[ak-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*_adims[i+1];
      asize=strides[0]*_adims[0];

      cst=roundup(asize,32); 
      memsize=2*cst; 
    }


    RtensorA(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, 
      const int _memsize, const int _cst, const int _dev=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), dev(_dev){

      if(dev==0){
        arr=new float[memsize];
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }

    }

    /*
    RtensorA(const int _k, const Gdims& _dims, const int _nbu, const vector<int>& _strides, const int _asize, 
      const int _memsize, const int _cst, const int _dev, const float* _arr, const float* _arrc, const view_flag& flag):
      k(_k), dims(_dims), nbu(_nbu), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), dev(_dev), 
      arr(_arr), arrc(_arrc), is_view(true){}
    */
    
    RtensorA(const int _k, const Gdims& _dims, const int _nbu, const vector<int>& _strides, const int _asize, 
      const int _memsize, const int _dev, float* _arr, const view_flag& flag):
      k(_k), dims(_dims), nbu(_nbu), strides(_strides), asize(_asize), memsize(_memsize), cst(_asize), dev(_dev), 
      arr(_arr), is_view(true){
    }
    

  public: // ---- Shape and strides -------------------------------------------------------------------------

    
    void reshape(const Gdims& _dims){
      assert(_dims.asize()==asize);
      dims=_dims;
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=cst; 
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


    RtensorA(const Gdims& _dims, const int _nbu, const int _dev):
      RtensorA(_dims.prepend(_nbu),_dev){
    }

    RtensorA(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      RtensorA(_dims,_dev){}
    
    RtensorA(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      RtensorA(_dims,_dev){
      if(dev==0) std::fill(arr,arr+asize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,asize*sizeof(float)));
    }

    RtensorA(const Gdims& _dims, const fill_ones& dummy, const int _dev=0): 
      RtensorA(_dims,fill::raw,0){
      std::fill(arr,arr+asize,1);
      if(_dev==1) move_to_device(_dev);
    }

    RtensorA(const Gdims& _dims, const fill_identity& dummy, const int _dev=0): 
      RtensorA(_dims,fill::raw,0){
      assert(dims[k-1]==dims[k-2]);
      std::fill(arr,arr+memsize,0);
      for(int i=0; i<dims[k-1]; i++)
	arr[i*(strides[k-2]+1)]=1;
      move_to_device(_dev);
    }

    RtensorA(const Gdims& _dims, const fill_gaussian& dummy, const int _dev):
      RtensorA(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      move_to_device(_dev);
    }

    RtensorA(const Gdims& _dims, const fill_gaussian& dummy, const float c, const int _dev):
      RtensorA(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=c*distr(rndGen);
      move_to_device(_dev);
    }

    RtensorA(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      RtensorA(_dims,fill::zero,0){
      for(int i=0; i<asize; i++) arr[i]=i;
      move_to_device(_dev);
    }
	  
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    RtensorA(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      RtensorA(_dims.prepend(_nbu),fill,_dev){
      nbu=_nbu;
    }
	  
    RtensorA(const Gdims& _dims, const int _nbu, const fill_gaussian& fill, const float c, const int _dev=0):
      RtensorA(_dims.prepend(_nbu),fill,c,_dev){
      nbu=_nbu;
    }


  public: // ---- Named constructors --------------------------------------------------------------------------


    static RtensorA noalloc(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_noalloc(),_dev);
    }

    static RtensorA raw(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_raw(),_dev);
    }

    static RtensorA zero(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_zero(),_dev);
    }

    static RtensorA zeros(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_zero(),_dev);
    }

    static RtensorA gaussian(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_gaussian(),_dev);
    }

    static RtensorA sequential(const Gdims& _dims, const int _dev=0){
      return RtensorA(_dims,fill_sequential(),_dev);
    }


  public: // ---- Lambda constructors -------------------------------------------------------------------------


    RtensorA(const Gdims& _dims, std::function<float(const int i, const int j)> fn):
      RtensorA(_dims,fill::raw,0){
      assert(dims.size()==2);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  set_value(i,j,fn(i,j));
    }

    RtensorA(const RtensorA& x, std::function<float(const float)> fn):
      RtensorA(x.dims,fill::raw,0){
      assert(x.dev==0);
      for(int i=0; i<asize; i++){
	float t=fn(x.arr[i]);
	arr[i]=std::real(t);
      }
    }

    RtensorA(const RtensorA& x, std::function<float(const int i, const int j, const float)> fn):
      RtensorA(x.dims,fill::raw,0){
      assert(x.dev==0);
      assert(x.dims.size()==2);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  set_value(i,j,fn(i,j,float(x(i,j))));
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    RtensorA(const RtensorA& x): 
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      CNINE_COPY_WARNING();
      if(dev==0){
        std::copy(x.arr,x.arr+asize,arr);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	      CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
#endif 
    }
        
    RtensorA(const RtensorA& x, const nowarn_flag& dummy): 
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
#endif 
    }
        
    RtensorA(const RtensorA& x, const int _dev): 
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,_dev){
      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("cnine error in RtensorA: device must be 0 or 1"));
      if(dev==0){
	if(x.dev==0){
	  std::copy(x.arr,x.arr+asize,arr);
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arr,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToHost)); 
	}
      }
      if(dev==1){
#ifdef _WITH_CUDA
	if(x.dev==0){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arr,asize*sizeof(float),cudaMemcpyHostToDevice));
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	}
#endif 
      }
    }

    RtensorA(const RtensorA& x, const view_flag& dummy){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev; 
      memsize=x.memsize; cst=x.cst; 
      arr=x.arr;
      arrg=x.arrg;
      is_view=true;
      //cout<<"RtensorA view "<<endl;
    }
        
    RtensorA(RtensorA&& x): 
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
      //cout<<"move RtensorA 0"<<endl; 
    }

    RtensorA(RtensorA& x, const string _dummy): // deprecated debugging only
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
      cout<<"move RtensorA 1"<<endl; 
    }

    RtensorA(const RtensorA& x, const fill_raw& dummy): 
      RtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){}

    RtensorA* clone() const{
      return new RtensorA(*this);
    }

    RtensorA& operator=(const RtensorA& x){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev;
      memsize=x.memsize; cst=x.cst;
      CNINE_ASSIGN_WARNING();

      if(is_view){
	if(dev==0){
	  std::copy(x.arr,x.arr+asize,arr);
	}
	if(dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	}
	return *this;
      }

      delete arr;
      if(arrg){CUDA_SAFE(cudaFree(arrg));}
      if(dev==0){
	arr=new float[memsize]; 
	std::copy(x.arr,x.arr+asize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      
      return *this;
    }


    RtensorA& operator=(RtensorA&& x){
      CNINE_MOVEASSIGN_WARNING();
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev; 
      memsize=x.memsize; cst=x.cst; 
      if(!is_view && arr) delete arr;
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Views -------------------------------------------------------------------------------


    RtensorA view(){
      RtensorA R=RtensorA::noalloc(dims,dev);
      R.arr=arr;
      R.is_view=true;
      return R;
    }


    RtensorA view_as_shape(const Gdims& _dims){
      CNINE_DIMS_EQ_TOTAL(dims,_dims);
      RtensorA R=RtensorA::noalloc(_dims,dev);
      R.arr=arr;
      R.is_view=true;
      return R;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    RtensorA(const Gtensor<float>& x, const int _dev=0): 
      RtensorA(x.dims,fill::raw){
      assert(dev==0);
      for(int i=0; i<asize; i++){
	arr[i]=x.arr[i];
      }
      move_to_device(_dev);
    }
    
    Gtensor<float> gtensor() const{
      if(dev>0) return RtensorA(*this,0).gtensor();
      Gtensor<float > R(dims,fill::raw);
      assert(dev==0);
      for(int i=0; i<asize; i++){
	R.arr[i]=arr[i];
      }
      return R;
    }


#ifdef _WITH_ATEN

    static bool is_viewable(const at::Tensor& T){
      return true;
    }  

    static bool is_regular(const at::Tensor& T){
      int k=T.dim();
      Gdims Tdims(k,fill_raw());
      for(int i=0; i<k ; i++)
	Tdims[i]=T.size(i);
      Gstrides Tstrides(Tdims,1);
      for(int i=0; i<k; i++)
	if(Tstrides[i]!=T.stride(i)) return false;
      return true;
    }

    RtensorA(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();

      //doesn't work
      //AT_DISPATCH_ALL_TYPES(T.scalar_type(), "float",[&](){
      //  cout<<"dioioa"<<endl;
      //});

      //if(typeid(T.scalar_type())!=at::ScalarType::Float){
      //auto Td=T.to(torch::kFloat32);
      //(*this)=RtensorA(Td);
      //return;
      //}

      T.contiguous();
      k=T.dim();
      dims=Gdims(k,fill_raw());
      for(int i=0; i<k ; i++){
	dims[i]=T.size(i);
      }
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=cst; 
      dev=T.type().is_cuda();

      bool reg=true;
      for(int i=0; i<k; i++)
	if(T.stride(i)!=strides[i]){reg=false; break;}
      if(!reg){
	CNINE_CPUONLY();
	auto src=T.data<float>();
	arr=new float[memsize];
	Gstrides sstrides(k,fill_raw());
	for(int i=0; i<k; i++) sstrides[i]=T.stride(i);
	for(int i=0; i<asize; i++)
	  arr[i]=src[sstrides.offs(i,strides)];
      }

      if(dev==0){
	arr=new float[memsize];
	std::copy(T.data<float>(),T.data<float>()+asize,arr);
	//std::copy(T.data<complex<float> >(),T.data<complex<float> >()+asize,arr);
	//memmove((void*)(arr),(void*)(T.data<float>()),asize*sizeof(float));
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,T.data<float>(),asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }

    }


    static RtensorA view(at::Tensor& T){
      T.contiguous();
      if(!is_regular(T)) return RtensorA(T);
      
      RtensorA R;
      R.k=T.dim();
      R.dims.resize(R.k);
      for(int i=0; i<R.k ; i++)
	R.dims[i]=T.size(i);
      R.strides.resize(R.k);
      for(int i=0; i<R.k; i++)
	R.strides[i]=T.stride(i);
      R.asize=R.strides[0]*R.dims[0]; // TODO
      R.cst=R.asize; 
      R.memsize=R.cst; 
      R.dev=T.type().is_cuda();

      R.is_view=true;
      if(R.dev==0){
	R.arr=T.data<float>();
      }
      
      if(R.dev==1){
	R.arrg=T.data<float>();
      }

      return R;
    }

    
    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      vector<int64_t> v(k); for(int i=0; i<2; i++) v[i]=dims[i];
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+asize,R.data<float>());
      return R;
    }


    at::Tensor move_to_torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      vector<int64_t> v(k); for(int i=0; i<2; i++) v[i]=dims[i];
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+asize,R.data<float>());
      return R;
    }


#endif 


  public: // ---- Transport -----------------------------------------------------------------------------------


    RtensorA& move_to_device(const int _dev){
      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("Cnine error in RtensorA: device must be 0 or 1"));

      if(_dev==0){
	if(dev==0) return *this;
 	delete[] arr;
	arr=new float[memsize];
	CUDA_SAFE(cudaMemcpy(arr,arrg,asize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<RtensorA*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,asize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<RtensorA*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    
    RtensorA& move_to(const device& _dev){
      return move_to_device(_dev.id());
    }
    
    RtensorA to(const device& _dev) const{
      return RtensorA(*this,_dev.id());
    }

    RtensorA to_device(const int _dev) const{
      cout<<"to("<<_dev<<")"<<endl;
      return RtensorA(*this,_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }

    Gdims get_dims() const{
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }

    RtensorA_accessor accessor(){
      return RtensorA_accessor(arr,strides);
    }

    int dim(const int i) const{
      return dims[i];
    }

    int combined_size(const int a, const int b) const{
      assert(b<=k);
      assert(a<=b);
      if(b>0 && strides[b-1]==0) return 0;
      if(a>0) return (strides[a-1])/(strides[b-1]);
      if(b>0) return asize/strides[b-1];
      return 1; 
    }


  public: // ---- Gindex case ---------


    float operator()(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }

    float get_value(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }
    
    void set_value(const Gindex& ix, const float& v){
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=v;
    }

    void inc(const Gindex& ix, const float& v){
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]+=v;
    }

    RscalarA get(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"RtensorA::get(...) not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return RscalarA(arr[t]);
    }
    
    void set(const Gindex& ix, const RscalarA& x){
      CNINE_ASSERT(dev==0,"RtensorA::get(...) not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=x.val;
    }

    float get_value_at(const int t) const{
      CNINE_ASSERT(dev==0,"RtensorA::get_value_at() not implemented for GPU.\n");
      return arr[t];
    }
    
    void set_value_at(const int t, const float& v){
      CNINE_ASSERT(dev==0,"RtensorA::set_value_at() not implemented for GPU.\n");
      arr[t]=v;
    }


  public: // ---- k=1 special cases ---- 


    float operator()(const int i0) const{
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      return arr[t];
    }

    float get_value(const int i0) const{
      CNINE_ASSERT(dev==0,"RtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      return arr[t];
    }

    void set_value(const int i0, const float x){
      CNINE_ASSERT(dev==0,"RtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      arr[t]=x;
    }

    void inc(const int i0, const float x){
      CNINE_ASSERT(dev==0,"RtensorA::inc not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]+=x;
    }

    RscalarA get(const int i0) const{
      CNINE_ASSERT(dev==0,"RtensorA::get(int) not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      return RscalarA(arr[i0]);
    }

    void set(const int i0, const RscalarA& x){
      CNINE_ASSERT(dev==0,"RtensorA::set not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]=x.val;
    }


  public: // ---- k=2 special cases ---- 


    float operator()(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"RtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return arr[t];
    }

    float get_value(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"RtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return arr[t];
    }

    void set_value(const int i0, const int i1, const float x){
      CNINE_ASSERT(dev==0,"RtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]=x;
    }

    void inc(const int i0, const int i1, const float x){
      CNINE_ASSERT(dev==0,"RtensorA::inc not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]+=x;
    }

    RscalarA get(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"RtensorA::get not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return RscalarA(arr[t]);
    }

    void set(const int i0, const int i1, const RscalarA& x){
      CNINE_ASSERT(dev==0,"RtensorA::set not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]=x.val;
    }


  public: // ---- k=3 special cases ----


    float operator()(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "RtensorA::operator() not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return arr[t];
    }

    float get_value(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "RtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return arr[t];
    }

    void set_value(const int i0, const int i1, const int i2, const float x){
      CNINE_ASSERT(dev==0, "RtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=x;
    }

    void inc(const int i0, const int i1, const int i2, const float x){
      CNINE_ASSERT(dev==0, "RtensorA::inc not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]+=x;
    }

    RscalarA get(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "RtensorA::get not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return RscalarA(arr[t]);
    }

    void set(const int i0, const int i1, const int i2, const RscalarA& x){
      CNINE_ASSERT(dev==0, "RtensorA::set not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x.val);
    }


  public: // ---- k=4 special cases ----


    float operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0, "RtensorA::operator() not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return arr[t];
    }

    float get_value(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0, "RtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return arr[t];
    }

    void set_value(const int i0, const int i1, const int i2, const int i3, const float x){
      CNINE_ASSERT(dev==0, "RtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      arr[t]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const float x){
      CNINE_ASSERT(dev==0, "RtensorA::inc not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      arr[t]+=x;
    }

    RscalarA get(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0, "RtensorA::get not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return RscalarA(arr[t]);
    }

    void set(const int i0, const int i1, const int i2, const int i3, const RscalarA& x){
      CNINE_ASSERT(dev==0, "RtensorA::set not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      arr[t]=std::real(x.val);
    }


    // ---- Cumulative 


    void add_element_into(RscalarA& r, const Gindex& ix){
      if(nbu==-1){
	r.val+=get_value(ix);
      }else{
	CNINE_UNIMPL();
      }
    }

    void add_to_element(const Gindex& ix, RscalarA& r){
      if(nbu==-1){
	inc(ix,r.val);
      }else{
	CNINE_UNIMPL();
      }
    }


  public: // ---- Access views --------------------------------------------------------------------------------


    RtensorA(const Rtensor1_view& x):
      RtensorA({x.n0},fill_zero(),x.dev){
      view1D().add(x);
    }
      //for(int i=0; i<x.n0; i++)
      //set(i,x(i));
      
    Rtensor1_view view1(){
      return Rtensor1_view(arr,dims,strides);
    }

    const Rtensor1_view view1() const{
      return Rtensor1_view(arr,dims,strides);
    }

    Rtensor1_view view1D(){
      return Rtensor1_view(arr,dims,strides);
    }

    const Rtensor1_view view1D() const{
      return Rtensor1_view(arr,dims,strides);
    }



    RtensorA(const Rtensor2_view& x):
      RtensorA({x.n0,x.n1},fill_zero(),x.dev){
      view2D().add(x);
    }

    Rtensor2_view view2(){
      return Rtensor2_view(arr,dims,strides);
    }

    const Rtensor2_view view2() const{
      return Rtensor2_view(arr,dims,strides);
    }

    Rtensor2_view view2D(){
      return Rtensor2_view(arr,dims,strides);
    }

    const Rtensor2_view view2D() const{
      return Rtensor2_view(arr,dims,strides);
    }

    //const Rtensor2_view view2D() const{
    //return Rtensor2_view(arr,dims,strides);
    //}

    Rtensor2_view view2(const GindexSet& a, const GindexSet& b){
      return Rtensor2_view(arr,dims,strides,a,b);
    }

    Rtensor2_view view2D(const GindexSet& a, const GindexSet& b){
      return Rtensor2_view(arr,dims,strides,a,b);
    }

    //Rtensor2_view view2D_block(const int i0, const int i1, const int n0, const int n1){
    //assert(dims.size()==2);
    //return Rtensor2_view(arr+i0*strides[0]+i1*strides[i1],n0,n1,strides[0],strides[1],dev);
    //}

    //const Rtensor2_view view2D_block(const int i0, const int i1, const int n0, const int n1) const{
    //assert(dims.size()==2);
    //return Rtensor2_view(arr+i0*strides[0]+i1*strides[i1],n0,n1,strides[0],strides[1],dev);
    //}

    Rtensor2_view block2(const int i0, const int i1, int m0=-1, int m1=-1){
      assert(dims.size()==2);
      if(m0==-1) m0=dims(0)-i0;
      if(m1==-1) m1=dims(1)-i1;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims(0) || i1>=dims(1)) 
	  throw std::out_of_range("cnine::RtensorA::block2: index "+Gindex({i0,i1}).str()+" out of range of size "+dims.str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i0+m0>dims(0) || i1+m1>dims(1)) 
	 throw std::out_of_range("cnine::RtensorA::block2: end index "+Gindex({i0+m0,i1+m1}).str()+" out of range of size "+dims.str()));
      return Rtensor2_view(arr+i0*strides[0]+i1*strides[1],m0,m1,strides[0],strides[1],dev);
    }

    const Rtensor2_view block2(const int i0, const int i1, int m0=-1, int m1=-1) const{
      assert(dims.size()==2);
      if(m0==-1) m0=dims(0)-i0;
      if(m1==-1) m1=dims(1)-i1;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims(0) || i1>=dims(1)) 
	  throw std::out_of_range("cnine::RtensorA::block2: index "+Gindex({i0,i1}).str()+" out of range of size "+dims.str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i0+m0>dims(0) || i1+m1>dims(1)) 
	 throw std::out_of_range("cnine::RtensorA::block2: end index "+Gindex({i0+m0,i1+m1}).str()+" out of range of size "+dims.str()));
      return Rtensor2_view(arr+i0*strides[0]+i1*strides[1],m0,m1,strides[0],strides[1],dev);
    }



    RtensorA(const Rtensor3_view& x):
      RtensorA({x.n0,x.n1,x.n2},fill_zero(),x.dev){
      view3D().add(x);
    }

    Rtensor3_view view3(){
      return Rtensor3_view(arr,dims,strides);
    }

    const Rtensor3_view view3() const{
      return Rtensor3_view(arr,dims,strides);
    }

    Rtensor3_view block3(const int i0, const int i1, const int i2, 
      int m0=-1, int m1=-1, int m2=-1){
      assert(dims.size()==3);
      if(m0==-1) m0=dims(0)-i0;
      if(m1==-1) m1=dims(1)-i1;
      if(m2==-1) m2=dims(2)-i2;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims(0) || i1>=dims(1) || i2>=dims(2)) 
	  throw std::out_of_range("cnine::RtensorA::block3: index "+Gindex({i0,i1,i2}).str()+" out of range of size "+dims.str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i2+m2<0 || i0+m0>dims(0) || i1+m1>dims(1)|| i2+m2>dims(2)) 
	  throw std::out_of_range("cnine::RtensorA::block3: end index "+Gindex({i0+m0,i1+m1,i2+m2}).str()+" out of range of size "+dims.str()));
      return Rtensor3_view(arr+i0*strides[0]+i1*strides[1]+i2*strides[2],m0,m1,m2,strides[0],strides[1],strides[2],dev);
    }

    const Rtensor3_view block3(const int i0, const int i1, const int i2, 
      int m0=-1, int m1=-1, int m2=-1) const{
      assert(dims.size()==3);
      if(m0==-1) m0=dims(0)-i0;
      if(m1==-1) m1=dims(1)-i1;
      if(m2==-1) m2=dims(2)-i2;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims(0) || i1>=dims(1) || i2>=dims(2)) 
	  throw std::out_of_range("cnine::RtensorA::block3: index "+Gindex({i0,i1,i2}).str()+" out of range of size "+dims.str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i2+m2<0 || i0+m0>dims(0) || i1+m1>dims(1)|| i2+m2>dims(2)) 
	  throw std::out_of_range("cnine::RtensorA::block3: end index "+Gindex({i0+m0,i1+m1,i2+m2}).str()+" out of range of size "+dims.str()));
      return Rtensor3_view(arr+i0*strides[0]+i1*strides[1]+i2*strides[2],m0,m1,m2,strides[0],strides[1],strides[2],dev);
    }

    Rtensor3_view view3D(){
      return Rtensor3_view(arr,dims,strides);
    }

    const Rtensor3_view view3D() const{
      return Rtensor3_view(arr,dims,strides);
    }



    RtensorA(const Rtensor4_view& x):
      RtensorA({x.n0,x.n1,x.n2,x.n3},fill_zero(),x.dev){
      view4D().add(x);
    }

    Rtensor4_view view4(){
      return Rtensor4_view(arr,dims,strides);
    }

    const Rtensor4_view view4() const{
      return Rtensor4_view(arr,dims,strides);
    }

    Rtensor4_view view4D(){
      return Rtensor4_view(arr,dims,strides);
    }

    const Rtensor4_view view4D() const{
      return Rtensor4_view(arr,dims,strides);
    }


    RtensorView viewx(){
      return RtensorView(arr,dims,strides);
    }

    const RtensorView viewx() const{
      return RtensorView(arr,dims,strides);
    }



    

  public: // ---- Chunks -------------------------------------------------------------------------------------


    void add_to_chunk(const int ix, const int offs, const RtensorA& x){
      assert(x.dev==dev);
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(dims[i]>=x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  //for(int m=0; m<x.dims[ix];; m++){
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]+=x.arr[j*subsize+i];
	  }
	  //toffs+=strides[ix];
	  //}
	}
	return; 
      }
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    }


    void set_chunk(const int ix, const int offs, const RtensorA& x){
      assert(x.dev==dev);
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(dims[i]>=x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  //for(int m=0; m<x.dims[ix];; m++){
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]=x.arr[j*subsize+i];
	  }
	  //toffs+=strides[ix];
	  //}
	}
	return; 
      }
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    }


    void add_chunk_of(const RtensorA& x, const int ix, const int offs, const int n){
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(x.dims[i]>=dims[i]);
      int subsize=strides[ix];
      int supsize=x.asize/(strides[ix]*dims[ix]);
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      int jxstride=x.asize;
      if(ix>0) jxstride=x.strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  for(int m=0; m<n; m++){
	    for(int i=0; i<subsize; i++){
	      arr[j*jstride+m*strides[ix]+i]+=x.arr[j*jxstride+(m+offs)*x.strides[ix]+i];
	    }
	  }
	}
	return; 
      }
      CNINE_UNIMPL();
    }


    RtensorA chunk(const int ix, const int offs, const int n=1) const{
      Gdims _dims(dims);
      _dims[ix]=n;
      RtensorA x(_dims,fill::raw,dev);
      int subsize=strides[ix];
      int supsize=asize/(strides[ix]*dims[ix]);
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      int jxstride=x.asize;
      if(ix>0) jxstride=x.strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  for(int m=0; m<n; m++){
	    for(int i=0; i<subsize; i++){
	      x.arr[j*jxstride+m*x.strides[ix]+i]=arr[j*jstride+(m+offs)*strides[ix]+i];
	    }
	  }
	}
	return x; 
      }
      CNINE_UNIMPL();
      return x; 
    }


  public: // ---- Slices -------------------------------------------------------------------------------------


    void add_to_slice(const int ix, const int offs, const RtensorA& x){
      assert(k==x.k+1);
      assert(x.dev==dev);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]+=x.arr[j*subsize+i];
	  }
	}
	return; 
      }
      CNINE_UNIMPL();
    }


    void add_to_slices(const int ix, const vector<const RtensorA*> v){
      assert(v.size()==dims[ix]);
      const RtensorA& x=*v[0];
      assert(k==x.k+1);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int m=0; m<dims[ix]; m++){
	  for(int j=0; j<supsize; j++){
	    int toffs=j*jstride+m*strides[ix];
	    const RtensorA& x=*v[m];
	    for(int i=0; i<subsize; i++){
	      arr[toffs+i]+=x.arr[j*subsize+i];
	    }
	  }
	}
	return; 
      }
      CNINE_UNIMPL();
    }


    void add_slice_of(const RtensorA& x, const int ix, const int offs){
      assert(x.dev==dev);
      assert(x.k==k+1);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<k; i++) assert(x.dims[i+1]==dims[i]);
      int subsize=asize;
      if(ix>0) subsize=strides[ix-1];
      int supsize=asize/subsize;
      int jstride=x.asize; 
      if(ix>0) jstride=x.strides[ix-1];
      
      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*x.strides[ix];
	  for(int i=0; i<subsize; i++){
	    arr[j*subsize+i]+=x.arr[toffs+i];
	  }
	}
	return; 
      }
      CNINE_UNIMPL();
    }


    RtensorA slice(const int ix, const int offs) const{
      RtensorA R(dims.remove(ix),fill::raw,dev);
      int subsize=R.asize;
      if(ix>0) subsize=R.strides[ix-1];
      int supsize=R.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      
      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  for(int i=0; i<subsize; i++){
	    R.arr[j*subsize+i]=arr[toffs+i];
	  }
	}
	return R; 
      }
      CNINE_UNIMPL();
      return R; 
    }


  public: // ---- In-place Operations ------------------------------------------------------------------------


    void set_zero(){
      if(dev==0){
	std::fill(arr,arr+asize,0);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg,0,asize*sizeof(float)));
      }
#endif
    }


    void inplace_times(const float c){
      if(dev==0){
	for(int i=0; i<asize; i++){
	  arr[i]*=c;
	}
	return; 
      }
      if(dev==1){
	const float cr = c;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, arrg, 1, arrg, 1));
      }
    }

    void inplace_times(const RscalarA& c){
      inplace_times(c.val);
    }

    void inplace_div(const RscalarA& c){
      inplace_times(float(1.0)/c.val);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    RtensorA conj() const{
      return *this;
    }

    RtensorA* conjp() const{
      return new RtensorA(*this);
    }

    RtensorA transp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	RtensorA R({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	  }
	return R;
      }
      RtensorA R(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      return R;
    }

    RtensorA* transpp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	RtensorA* R=new RtensorA({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R->arr[i*J+j]=arr[j*I+i];
	  }
	return R;
      }
      RtensorA* R=new RtensorA(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R->arrg,J,R->arrg,J));
      return R;
    }

    RtensorA plus(const RtensorA& x) const{
      RtensorA R(*this);
      cout<<"plus"<<endl;
      R.add(x);
      return R;
    }


    /*
    RtensorA* conj() const{
      return new RtensorA(CFtensor::conj());
    }

    RtensorA* transp() const{
      return new RtensorA(CFtensor::transp());
    }

    RtensorA* herm() const{
      return new RtensorA(CFtensor::herm());
    }
    */
    
    
    RtensorA* divide_colsp(const RtensorA& N) const{
      return new RtensorA(divide_cols(N));
    }

    RtensorA* normalize_colsp() const{
      return new RtensorA(normalize_cols());
    }

    RtensorA divide_cols(const RtensorA& N) const{
      assert(N.dev==dev);
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      RtensorA R(dims,fill::zero,0);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      R.arr[offs+i*J+j]/=z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
      return R;
    }

    RtensorA normalize_cols() const{
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      RtensorA R(*this);
      if(dev==0){
	for(int i=0; i<I; i++){
	  float tr=0;
	  for(int j=0; j<J; j++){
	    tr+=R.arr[i*J+j]*R.arr[i*J+j];
	  }
	  float z=sqrt(tr);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
      return R;
    }

    
    RtensorA tensor_product(const RtensorA& y){
      RtensorA R(dims.cat(y.dims),fill_raw());
      assert(dims.size()==2);
      assert(y.dims.size()==2);
      for(int a=0; a<dims(0); a++)
	for(int b=0; b<dims(1); b++)
	  for(int c=0; c<y.dims(0); c++)
	    for(int d=0; d<y.dims(1); d++)
	      R.set_value(a,b,c,d,(*this)(a,b)*y(c,d));
      return R;
    }


  public: // ---- Scalar valued operations ------------------------------------------------------------------


    float norm2() const{
      if(dev==0){
      float t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*arr[i];
      return t;
      }
      float t=0;
      CUBLAS_SAFE(cublasSdot(cnine_cublas, asize, arrg, 1, arrg, 1, &t));
      return t;
    }


    float inp(const RtensorA& x) const{
      assert(asize==x.asize);
      if(asize==0) return 0; 
      assert(x.dev==dev);
      if(dev==0){
	float tr=0; 
	for(int i=0; i<asize; i++){
	  tr+=arr[i]*x.arr[i];
	}
	return tr;
      }
      float a=0;
      CUBLAS_SAFE(cublasSdot(cnine_cublas, asize, arrg, 1, x.arrg, 1, &a));
      return a;
    }


    float diff2(const RtensorA& x) const{
      CNINE_DEVICE_SAME(x);
      assert(asize==x.asize);
      if(asize==0) return 0; 
      if(dev==0){
	float t=0; 
	for(int i=0; i<asize; i++){
	  float a=arr[i]-x.arr[i];
	  t+=a*a;
	}
	return t;
      }
      float a=0;
      CNINE_CPUONLY();
      //CUBLAS_SAFE(cublasSdot(cnine_cublas, asize, arrg, 1, x.arrg, 1, &a));
      return a;
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void set(const RtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr);
	return; 
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }


    void add(const RtensorA& x){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      if(dev==0){
        for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	      return; 
      }
      if(dev==1){
        cout<<this->str()<<endl;
        cout<<"GPU add"<<asize<<endl;
	      const float alpha = 1.0;
	      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      }
    }


    void operator+=(const RtensorA& x){
      add(x);
    }

    void add(const RtensorA& x, const float c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
	return;
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrg, 1, arrg, 1));
      }
    }


    void add(const RtensorA& x, const RscalarA& c){
      assert(c.nbu==-1);
      add(x,c.val);
    }

    void add(const Rtensor1_view& x){
      view1().add(x);
    }

    void add(const Rtensor2_view& x){
      view2().add(x);
    }

    void add(const Rtensor3_view& x){
      view3().add(x);
    }

    void add(const Rtensor4_view& x){
      view4().add(x);
    }

    void add_prod(const RscalarA& c, const RtensorA& A){
      assert(c.nbu==-1);
      add(A,c.val);
    }
 
    void add_divide(const RtensorA& x, const float c){
      add(x,1.0/c);
    }

    void add_divide(const RtensorA& x, const RscalarA& c){
      assert(c.nbu==-1);
      add(x,1.0/c.val);
    }


    void add_sum(const vector<RtensorA*> v){
      const int N=v.size();
      if(N==0) return; 
      if(dev==0){
	for(int i=0; i<N; i++){
	  const RtensorA& o=*v[i];
	  assert(o.asize==asize);
	  assert(o.dev==dev);
	  for(int j=0; j<asize; j++){
	    arr[j]+=o.arr[j];
	  }
	}
	return;
      }
      const float alpha = 1.0;
      for(int i=0; i<N; i++){
	const RtensorA& o=*v[i];
	assert(o.asize==asize);
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, o.arrg, 1, arrg, 1));
	//cudaDeviceSynchronize();
      }
    }

    void subtract(const RtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.dev==dev);
      assert(asize==x.asize);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
	return;
      }
      const float c=-1.0; 
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrg, 1, arrg, 1));
    }

    void add_plus(const RtensorA& x, const RtensorA& y){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(y.dims));
      assert(asize==x.asize);
      assert(asize==y.asize);
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]+y.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, y.arrg, 1, arrg, 1));
      }
    }

    void add_minus(const RtensorA& x, const RtensorA& y){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(y.dims));
      assert(asize==x.asize);
      assert(asize==y.asize);
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]-y.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	const float malpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, y.arrg, 1, arrg, 1));
      }
    }

    void add_transp(const RtensorA& x, const int n=1) const{
      CNINE_CHECK_SIZE(dims.check_eq(x.dims.transpose()));
      assert(asize==x.asize);
      assert(x.dev==dev);
      const int J=x.combined_size(0,n);
      const int I=x.asize/J;
      if(dev==0){
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    arr[i*J+j]+=x.arr[j*I+i];
	  }
	return;
      }
      if(dev==1){
	const float alpha = 1.0;
	const float beta = 1.0;
	CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
      }
    }


  public: // ---- Into Operations ----------------------------------------------------------------------------


    void add_norm2_into(RscalarA& r) const{
      if(nbu==-1){
	r.val+=inp(*this);
      }else{
	CNINE_UNIMPL();
      }
    }

    void add_inp_into(RscalarA& r, const RtensorA& A) const{
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      if(nbu==-1){
	r.val+=inp(A);
      }else{
	CNINE_UNIMPL();
      }
    }


  public: // ---- Normalization ------------------------------------------------------------------------------


    void add_col_norms(const RtensorA& x){
      assert(x.dev==dev);
      int xk=x.dims.size();
      assert(xk>=2);
      const int J=x.dims[xk-1];
      const int I=x.dims[xk-2];
      const int A=x.asize/(I*J);
      assert(asize==A*J);

      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float t=0;
	    for(int i=0; i<I; i++){
	      t+=x.arr[offs+i*J+j]*x.arr[offs+i*J+j];
	    }
	    arr[a*J+j]+=sqrt(t);
	  }
	}
	return;
      }
      CNINE_UNIMPL();
    }


    void add_col_norms_back(const RtensorA& G, const RtensorA& X, const RtensorA& N){
      assert(G.dev==dev);
      assert(X.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      assert(X.asize==asize);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      assert(G.asize==N.asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=G.arr[a*J+j]/N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      arr[offs+i*J+j]+=X.arr[offs+i*J+j]*z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols(const RtensorA& X, const RtensorA& N){
      assert(X.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      assert(X.asize==asize);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    //float z=float(N.arr[a*J+j],N.arrc[a*J+j]);
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      //float u=float()
	      arr[offs+i*J+j]+=X.arr[offs+i*J+j]/z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols_back0(const RtensorA& G, const RtensorA& N){
      assert(G.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      assert(G.asize==asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float n=N.arr[a*J+j]; //-N.arrc[a*J+j]);
	    //float z=float(1,0)/n/n; //float(G.arr[a*J+j],G.arrc[a*J+j])/n/n;
	    for(int i=0; i<I; i++){
	      float u=G.arr[offs+i*J+j]/n;
	      //float u=z*float(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
	      //float(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j]);
	      arr[offs+i*J+j]+=std::real(u);
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols_back1(const RtensorA& G, const RtensorA& X, const RtensorA& N){
      assert(G.dev==dev);
      assert(X.dev==dev);
      assert(N.dev==dev);
      const int _k=G.k;
      assert(_k>=2);
      assert(G.dims==X.dims);
      assert(dims==N.dims);
      const int J=G.dims[_k-1];
      const int I=G.dims[_k-2];
      const int A=G.asize/(I*J);
      assert(N.asize==G.asize/I);
      assert(asize==N.asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=-pow(N.arr[a*J+j],-2);
	    for(int i=0; i<I; i++){ // improve
	      float t=G.arr[offs+i*J+j]*X.arr[offs+i*J+j]*z;
	      arr[a*J+j]+=std::real(t);
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }

  protected: // ---- Matrix multiplication -------------------------------------------------------------------
  public:
    
    // The last nx indices of x are contracted with the first ny indices of y
    // Selector: x is conjugated if selector is 1 or 3
    // Selector: y is conjugated if selector is 2 or 3

    void add_Mprod_AA(const RtensorA& x, const RtensorA& y, const int nx=1, const int ny=1){

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      //cout<<dims<<"="<<x.dims<<"*"<<y.dims<<endl;
      assert(y.combined_size(0,ny)==K);

      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(ny,y.k);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int pstridey=J;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    float ti=0;
	    for(int p=0; p<K; p++){
	      int qx=i*istridex+p;
	      int qy=p*pstridey+j;
	      float xr=x.arr[qx];
	      float yr=y.arr[qy];
	      tr+=xr*yr;
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	  }
      }

      if(dev>0){

	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,J,x.arrg,K,&beta,arrg,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


    // The last nx indices of x are contracted with the last ny indices of y
    void add_Mprod_AT(const RtensorA& x, const RtensorA& y, const int nx=1, const int ny=1){

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      assert(y.combined_size(y.k-ny,y.k)==K);

      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(0,y.k-ny);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int jstridey=K;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    for(int p=0; p<K; p++){
	      int qx=i*istridex+p;
	      int qy=p+j*jstridey;
	      float xr=x.arr[qx];
	      float yr=y.arr[qy];
	      tr+=xr*yr;
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	  }

      }

      if(dev>0){

	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,K,x.arrg,K,&beta,arrg,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


    // The first nx indices of x are contracted with the first ny indices of y
    void add_Mprod_TA(const RtensorA& x, const RtensorA& y, const int nx=1, const int ny=1){
  
      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(0,nx);
  
      assert(y.combined_size(0,ny)==K);

      const int I=x.combined_size(nx,x.k);
      const int J=y.combined_size(ny,y.k);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istrider=J;
	const int pstridex=I;
	const int pstridey=J;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    for(int p=0; p<K; p++){
	      int qx=i+p*pstridex;
	      int qy=p*pstridey+j;
	      float xr=x.arr[qx];
	      float yr=y.arr[qy];
	      tr+=xr*yr;
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	  }

      }

      if(dev>0){
	
	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha0,
	    y.arrg,J,x.arrg,I,&beta,arrg,J)); 
	//cudaDeviceSynchronize(); 
      }
      
    }

  public: // ---- Special functions --------------------------------------------------------------------------


    void add_ReLU(const RtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*x.arr[i];
    }

    void add_ReLU(const RtensorA& x, const float alpha){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
    }

    void add_ReLU_back(const RtensorA& g, const RtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(g.asize==asize);
      assert(x.dev==dev);
      assert(g.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*g.arr[i];
    }

    void add_ReLU_back(const RtensorA& g, const RtensorA& x, const float alpha){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(g.asize==asize);
      assert(x.dev==dev);
      assert(g.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+(x.arr[i]<=0)*alpha)*g.arr[i];
    }

    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    string str(const string indent, const float eps) const{
      return gtensor().str(indent,eps);
    }

    friend ostream& operator<<(ostream& stream, const RtensorA& x){
      stream<<x.str(); return stream;}
   
  };


  // ---- Post-class inline functions ------------------------------------------------------------------------




}

#endif

