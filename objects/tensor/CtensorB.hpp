//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensorB
#define _CnineCtensorB

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "CtensorB_accessor.hpp"

#include "Ctensor1_view.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "CtensorD_view.hpp"

#include "CtensorView.hpp"
#include "Aggregator.hpp"

#include "Ctensor_mprodFn.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  class CtensorB{
  public:

    Gdims dims;
    Gstrides strides;

    int asize=0;
    int memsize=0;
    int coffs=0; 

    int dev=0;
    bool is_view=false;

    float* arr=nullptr;
    float* arrg=nullptr;


  public:

    CtensorB(){}

    ~CtensorB(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const{
      return "CtensorB";
    }

    string describe() const{
      return "CtensorB"+dims.str();
    }


  public: // ---- Constructors -----------------------------------------------------------------------------


    CtensorB(const Gdims& _dims, const Gstrides& _strides, 
      const int _asize, const int _memsize, const int _coffs, const int _dev):
     dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), coffs(_coffs), dev(_dev){}


    CtensorB(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims,2){
      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("Cnine error in CtensorB: device must be 0 or 1"));
      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;
    }


    // Root constructor: elements are uninitialized
    CtensorB(const Gdims& _dims, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims,2){

      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("Cnine error in CtensorB: device must be 0 or 1"));

      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;

      if(dev==0){
	arr=new float[memsize];
      }

      if(dev==1){
	CNINE_REQUIRES_CUDA()
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }

    }


  public: // ---- Filled constructors -----------------------------------------------------------------------

    
    //CtensorB(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0):
    //CtensorB(_dims,_dev){}

    CtensorB(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      CtensorB(_dims,_dev){}
    
    CtensorB(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      CtensorB(_dims,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CtensorB(const Gdims& _dims, const fill_ones& dummy, const int _dev=0): 
      CtensorB(_dims,fill::raw){
      std::fill(arr,arr+memsize,1);
      if(_dev==1) move_to_device(_dev);
    }

    CtensorB(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      CtensorB(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<memsize; i++) arr[i]=distr(rndGen);
      if(_dev==1) move_to_device(_dev);
    }
    
    /*
    CtensorB(const Gdims& _dims, const fill_gaussian& dummy, const float c, const int _dev):
      CtensorB(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<memsize; i++) arr[i]=c*distr(rndGen);
      move_to_device(_dev);
    }
    */

    CtensorB(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      CtensorB(_dims,fill::zero,0){
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++) arr[i*s]=i;
      //for(int i=0; i<asize; i++) arr[i*s+coffs]=0;
      move_to_device(_dev);
    }
	  

  public: // ---- Named constructors -------------------------------------------------------------------------


    static CtensorB raw(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_raw(),_dev);
    }

    static CtensorB zero(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_zero(),_dev);
    }

    static CtensorB zeros(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_zero(),_dev);
    }

    static CtensorB ones(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_ones(),_dev);
    }

    static CtensorB sequential(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_sequential(),_dev);
    }

    static CtensorB gaussian(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_gaussian(),_dev);
    }


    static CtensorB zeros_like(const CtensorB& x){
      return CtensorB(x.dims,fill_zero(),x.dev);
    }



  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorB(const CtensorB& x): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      CNINE_COPY_WARNING();
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	#ifdef _WITH_CUDA
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
	#endif 
      }
    }
        
    CtensorB(const CtensorB& x, const nowarn_flag& dummy): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	#ifdef _WITH_CUDA
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
	#endif 
      }
    }
        
    CtensorB(const CtensorB& x, const int _dev): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,_dev){
      if(dev==0){
	if(x.dev==0){
	  arr=new float[memsize];
	  std::copy(x.arr,x.arr+memsize,arr);
	}
	if(x.dev==1){
	  CNINE_REQUIRES_CUDA();
	  arr=new float[memsize];
	  CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
	}
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	#ifdef _WITH_CUDA
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	if(x.dev==0){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	}
	#endif 
      }
    }

    CtensorB(const CtensorB& x, const view_flag& dummy):
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      arr=x.arr;
      arrg=x.arrg;
      is_view=true;
    }
        
    CtensorB(CtensorB&& x): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
      //cout<<"move CtensorB "<<endl; 
    }

    CtensorB* clone() const{
      return new CtensorB(*this);
    }

    CtensorB& operator=(const CtensorB& x){
      CNINE_ASSIGN_WARNING();
      dims=x.dims; strides=x.strides; 
      asize=x.asize; 
      memsize=x.memsize; 
      coffs=x.coffs;
      dev=x.dev;

      /*
      if(is_view){
	if(dev==0){
	  std::copy(x.arr,x.arr+memsize,arr);
	}
	if(dev==1){
	  CNINE_REQUIRES_CUDA();
	  #ifdef _WITH_CUDA
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  #endif 
	}
	return *this;
      }
      */

      if(!is_view){
	if(arr){delete arr; arr=nullptr;}
	#ifdef _WITH_CUDA
	if(arrg){CUDA_SAFE(cudaFree(arrg)); arrg=nullptr;}
	#endif
      }

      if(dev==0){
	arr=new float[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	#ifdef _WITH_CUDA
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	#endif 
      }
      
      is_view=false;
      return *this;
    }


    CtensorB& operator=(CtensorB&& x){
      CNINE_MOVEASSIGN_WARNING();
      dims=x.dims; strides=x.strides; 
      asize=x.asize; 
      memsize=x.memsize; 
      coffs=x.coffs; 
      dev=x.dev; 
      if(!is_view && arr) {delete arr; arr=nullptr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg)); arrg=nullptr;}
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


  public: // ---- Transport -----------------------------------------------------------------------------------


    CtensorB& move_to_device(const int _dev){

      if(_dev==0){
	if(dev==0) return *this;
 	delete[] arr;
	arr=new float[memsize];
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<CtensorB*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CtensorB*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    
    CtensorB& move_to(const device& _dev){
      return move_to_device(_dev.id());
    }
    
    CtensorB to(const device& _dev) const{
      return CtensorB(*this,_dev.id());
    }

    CtensorB to_device(const int _dev) const{
      return CtensorB(*this,_dev);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorB(const Gtensor<complex<float> >& x, const int _dev=0): 
      CtensorB(x.dims,fill::raw){
      assert(x.dev==0);
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++){
	arr[s*i]=std::real(x.arr[i]);
	arr[s*i+coffs]=std::imag(x.arr[i]);
      }
      move_to_device(_dev);
    }
    
    Gtensor<complex<float> > gtensor() const{
      if(dev>0) return CtensorB(*this,0).gtensor();
      Gtensor<complex<float> > R(dims,fill::raw);
      assert(dev==0);
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++){
	R.arr[i]=complex<float>(arr[s*i],arr[s*i+coffs]);
      }
      return R;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    static bool is_regular(const at::Tensor& T){
      int k=T.dim();
      Gdims Tdims(k,fill_raw());
      for(int i=0; i<k ; i++)
	Tdims[i]=T.size(i);
      Gstrides Tstrides(Tdims,1);
      bool t=true;
      for(int i=0; i<k; i++)
	if(Tstrides[i]!=T.stride(i)) {t=false; break;}
      if(t==false){
	CoutLock lk;
	cout<<"Warning: ATen tensor of dims "<<Tdims<<" has strides [ ";
	for(int i=0; i<k; i++) cout<<T.stride(i)<<" ";
	cout<<"] instead of "<<Tstrides<<endl;
	return false; 
      }
      return true;
    }


    CtensorB(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      assert(typeid(T.type().scalarType())==typeid(float));

      T.contiguous();
      int k=T.dim()-1;
      if(k<=0 || T.size(k)!=2) throw std::out_of_range("CtensorB: last dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      dims=Gdims(k,fill_raw());
      for(int i=0; i<k ; i++){
	dims[i]=T.size(i);
      }
      strides=Gstrides(dims,2);
      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;
      dev=T.type().is_cuda();

      bool reg=true;
      for(int i=0; i<k; i++)
	if(T.stride(i)!=strides[i]){reg=false; break;}
      if(T.stride(k)!=2) reg=false;
      if(!reg){
	CNINE_CPUONLY();
	auto src=T.data<float>();
	arr=new float[memsize];
	Gstrides sstrides(k+1,fill_raw());
	Gstrides xstrides(strides); 
	xstrides.push_back(2);
	for(int i=0; i<k+1; i++) sstrides[i]=T.stride(i);
	for(int i=0; i<memsize; i++)
	  arr[i]=src[sstrides.offs(i,xstrides)];
      }

      if(dev==0){
	arr=new float[memsize];
	//std::copy(T.data<complex<float> >(),T.data<complex<float> >()+asize,reinterpret_cast<complex<float>* >(arr));
	std::copy(T.data<float>(),T.data<float>()+memsize,arr);
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,T.data<float>(),memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      //cout<<*this<<endl;
    }

    static CtensorB view(at::Tensor& T){
      T.contiguous();
      if(!is_regular(T)){cout<<"irregular!"<<endl; return CtensorB(T);}
      
      CtensorB R;
      int k=T.dim()-1;
      if(k<=0 || T.size(k)!=2) throw std::out_of_range("CtensorB: last dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      R.dims.resize(k);
      for(int i=0; i<k ; i++)
	R.dims[i]=T.size(i);
      R.strides=Gstrides(R.dims,2);
      R.asize=R.strides[0]*R.dims[0]/2; 
      R.memsize=R.strides[0]*R.dims[0]; 
      R.coffs=1;
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

    static CtensorB* viewp(at::Tensor& T){
      T.contiguous();
      if(!is_regular(T)){cout<<"irregular!"<<endl; return new CtensorB(T);}
      
      CtensorB* R=new CtensorB();
      int k=T.dim()-1;
      if(k<=0 || T.size(k)!=2) throw std::out_of_range("CtensorB: last dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      R->dims.resize(k);
      for(int i=0; i<k ; i++)
	R->dims[i]=T.size(i);
      R->strides=Gstrides(R->dims,2);
      R->asize=R->strides[0]*R->dims[0]/2; 
      R->memsize=R->strides[0]*R->dims[0]; 
      R->coffs=1;
      R->dev=T.type().is_cuda();
      R->is_view=true;

      if(R->dev==0){
	R->arr=T.data<float>();
      }
      
      if(R->dev==1){
	R->arrg=T.data<float>();
      }

      return R;
    }

    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      assert(coffs==1);
      int k=getk();
      vector<int64_t> v(k+1); 
      for(int i=0; i<k; i++) v[i]=dims[i];
      v[k]=2;
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+memsize,R.data<float>());
      return R;
    }

    /*
    at::Tensor move_to_torch(){ // TODO 
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      vector<int64_t> v(k+1); 
      for(int i=0; i<k; i++) v[i+1]=dims[i];
      v[0]=2;
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+asize,R.data<float>());
      std::copy(arrc,arrc+asize,R.data<float>()+asize);
      return R;
    }
    */

#endif 



  public: // ---- Access -------------------------------------------------------------------------------------


    int getk() const{
      return dims.size();
    }

    int ndims() const{
      return dims.size();
    }

    Gdims get_dims() const{
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }

    float* true_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }

  public: // ---- Accessors ----------------------------------------------------------------------------------

    
    //CtensorB_accessor1 access_as_1D() const{
    //return CtensorB_accessor1(strides);
    //}

    //CtensorB_accessor2 access_as_2D() const{
    //return CtensorB_accessor2(strides);
    //}

    //CtensorB_accessor3 access_as_3D() const{
    //return CtensorB_accessor3(strides);
    //}


  public: // ---- Access views --------------------------------------------------------------------------------


    Ctensor1_view view1() const{
      if(dev==0) return Ctensor1_view(arr,dims,strides,coffs,dev);
      else return Ctensor1_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor1_view view1D() const{
      return Ctensor1_view(arr,dims,strides,coffs,dev);
    }

    Ctensor1_view view1D(const GindexSet& a) const{
      return Ctensor1_view(arr,dims,strides,a,coffs);
    }


    Ctensor2_view view2() const{
      if(dev==0) return Ctensor2_view(arr,dims,strides,coffs,dev);
      else return Ctensor2_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor2_view view2D() const{
      return Ctensor2_view(arr,dims,strides,coffs,dev);
    }

    Ctensor2_view view2D(const GindexSet& a, const GindexSet& b) const{
      return Ctensor2_view(arr,dims,strides,a,b,coffs);
    }


    Ctensor3_view view3() const{
     if(dev==0) return Ctensor3_view(arr,dims,strides,coffs,dev);
     else return Ctensor3_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor3_view view3D() const{
      return Ctensor3_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor3_view view3D(const GindexSet& a, const GindexSet& b, const GindexSet& c) const{
      return Ctensor3_view(arr,dims,strides,a,b,c,coffs);
    }


    const Ctensor4_view view4() const{
      return Ctensor4_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor4_view view4(){
      return Ctensor4_view(true_arr(),dims,strides,coffs,dev);
    }


    CtensorView viewx(){
      return CtensorView(arr,arr+coffs,dims,strides);
    }

    const CtensorView viewx() const{
      return CtensorView(arr,arr+coffs,dims,strides);
    }


    Ctensor2_view pick_dimension(const int ix=0){
      int k=getk();
      assert(k>=2);
      Ctensor2_view r;

      if(ix==0){
	r.n0=dims(0);
	r.n1=strides[0]/strides[k-1];
	r.s0=strides[0];
	r.s1=strides[k-1];
	r.arr=arr;
	r.arrc=arr+coffs;
	return r;
      }

      if(ix==k-1){
	r.n0=dims(k-1);
	r.n1=asize/dims(k-1);
	r.s0=strides[k-1];
	r.s1=strides[k-2];
	r.arr=arr;
	r.arrc=arr+coffs;
	return r;
      }

      assert(false);
      return r;
    }

    
  public: // ---- Element Access ------------------------------------------------------------------------------
    

    complex<float> operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return complex<float>(arr[t],arr[t+coffs]);
    }


    void set(const int i0, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }


  public: // ---- Cells --------------------------------------------------------------------------------------


    CtensorB view_of_cell(const Gindex& cix){
      assert(coffs==1);
      assert(cix.size()<dims.size());
      CtensorB R(dims.chunk(cix.size()),fill_noalloc(),dev);
      R.arr=arr+cix(strides);
      R.is_view=true;
      return R;
    }

    const CtensorB view_of_cell(const Gindex& cix) const{
      assert(coffs==1);
      assert(cix.size()<dims.size());
      CtensorB R(dims.chunk(cix.size()),fill_noalloc(),dev);
      R.arr=arr+cix(strides);
      R.is_view=true;
      return R;
    }

    CtensorB get_cell(const Gindex& cix) const{
      return CtensorB(view_of_cell(cix),nowarn_flag());
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorB operator+(const CtensorB& y) const{
      CtensorB R(*this);
      R.add(y);
      return R;
    }

    CtensorB operator-(const CtensorB& y) const{
      CtensorB R(*this);
      R.subtract(y);
      return R;
    }

    CtensorB operator/(const CtensorB& y) const{ // TODO
      CNINE_CPUONLY();
      assert(y.memsize==memsize);
      CtensorB R=zeros_like(*this);
      for(int i=0; i<asize; i++){
	complex<float> t=complex<float>(arr[i*2],arr[i*2+coffs])/complex<float>(y.arr[i*2],y.arr[i*2+coffs]);
	R.arr[i*2]=std::real(t);
	R.arr[i*2+coffs]=std::imag(t);
      }
      for(int i=0; i<memsize; i++){
	R.arr[i]=arr[i]/y.arr[i];
      }
      return R;
    }

    
    float norm2() const{
      assert(dev==0);
      float t=0;
      for(int i=0; i<memsize; i++){
	t+=arr[i]*arr[i];
      }
      return t;
    }

    float diff2(const CtensorB& y) const{
      assert(dev==0);
      assert(memsize==y.memsize);
      float t=0;
      for(int i=0; i<memsize; i++){
	float d=arr[i]-y.arr[i];
	t+=d*d;
      }
      return t;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorB& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]+=x.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &alpha, x.arrg, 1, arrg, 1));
      }
    }

    void operator+=(const CtensorB& x){
      add(x);
    }

    void add(const CtensorB& x, const float c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]+=x.arr[i]*c;
	return;
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &c, x.arrg, 1, arrg, 1));
      }
    }


    void add(const CtensorB& x, const complex<float> c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      float cr=std::real(c);
      float ci=std::imag(c);
      if(dev==0){
	//for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
	//for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mci=-ci; 
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
      }
    }


    void subtract(const CtensorB& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]-=x.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &alpha, x.arrg, 1, arrg, 1));
      }
    }

    void operator-=(const CtensorB& x){
      subtract(x);
    }



    /*
    void add_matmul(const CtensorB& _x, const CtensorB& M, const int d){
      assert(M.getk()==2);
      assert(_x.getk()>d);
      assert(_x.dims(d)==M.dims(1));

      auto x=_x.like_matrix();
      auto r=this->like_matrix();

      int I=x.dims(1);
      int J=x.dims(0);
      for(int i=0; i<I; i++){
	for(int K=0; k<K; k++){
	  complex<float> t=0;
	  for(int j=0; j<J; j++){
	    //t+=x(i,j)*

	      //for(int i=0; i<r.dims(0); i++)
	  }
	}
      }
    }
    */

      
    void add_gather(const CtensorB& x, const Rmask1& mask){
      int k=dims.size();
      assert(x.dims.size()==k);
      if(k==2) Aggregator(view2(),x.view2(),mask);
      if(k==3) Aggregator(view3(),x.view3(),mask);
      if(k==4) Aggregator(view4(),x.view4(),mask);
    }


	

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    string str_as_array(const int k, const string indent="") const{
      ostringstream oss;
      assert(k<dims.size());
      Gdims arraydims=dims.chunk(0,k);
      arraydims.foreach_index([&](const vector<int>& ix){
	  oss<<indent<<"Cell"<<Gindex(ix)<<endl;
	  oss<<get_cell(ix).str(indent)<<endl;
	});
      return oss.str();
    }

    string repr() const{
      return "<cnine::CtensorB"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorB& x){
      stream<<x.str(); return stream;}
   







  };

}


#endif 
