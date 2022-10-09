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

#ifndef _RtensorPack
#define _RtensorPack

#include "array_pool.hpp"
#include "RtensorA.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "IntTensor.hpp"


namespace cnine{

  class RtensorPack{
  public:

    //typedef cnine::Gdims Gdims;
    typedef RtensorA rtensor;
    //typedef cnine::IntTensor IntTensor;
    //typedef cnine::Rtensor1_view Rtensor1_view;
    //typedef cnine::Rtensor2_view Rtensor2_view;
    //typedef cnine::Rtensor3_view Rtensor3_view;

    float* arr=nullptr;
    float* arrg=nullptr;
    int dev=0;
    int memsize=0;
    int tail=0;
    IntTensor dir;
    mutable IntTensor* dirg=nullptr;
    //bool is_view=false;


    ~RtensorPack(){
      //if(is_view) return;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
      if(dirg) delete dirg;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPack(){}

    RtensorPack(const int ndims, const int _dev):
      dev(_dev), dir(Gdims(0,ndims+1),cnine::fill_noalloc()){}

    RtensorPack(const IntTensor& _dir, const int _dev):
      dev(_dev), dir(_dir){}

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_raw& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_zero& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1){}
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_gaussian& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      CNINE_CPUONLY();
      int asize=_dims.asize();
      reserve(_N*asize);
      normal_distribution<double> distr;
      for(int i=0; i<_N*asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<_N; i++){
	dir.push_back(i*asize,_dims);
      }
      tail=_N*asize;
    }

    RtensorPack(const cnine::array_pool<int>& dimensions, const cnine::fill_zero& dummy, const int _dev=0){
      dev=_dev;
      dir=IntTensor(Gdims(0,dimensions(0).size()+1),cnine::fill_noalloc());

      int reserve_size=0;
      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	reserve_size+=t;
      }
      reserve(reserve_size);
      if(dev==0) std::fill(arr,arr+reserve_size,0);
      if(dev==1){}

      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	dir.push_back(tail,v);
	tail+=t;
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorPack zeros_like(const RtensorPack& x){
      RtensorPack R(x.dir,x.dev);
      R.reserve(x.tail);
      if(x.dev==0) std::fill(R.arr,R.arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R.arrg,0,R.tail*sizeof(float)));
      R.tail=x.tail;
      return R;
    }

    static RtensorPack* new_zeros_like(const RtensorPack& x){
      RtensorPack*  R=new RtensorPack(x.dir,x.dev);
      R->reserve(x.tail);
      if(x.dev==0) std::fill(R->arr,R->arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R->arrg,0,R->tail*sizeof(float)));
      R->tail=x.tail;
      return R;
    }


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	float* newarr=new float[newsize];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
      dirg_refresh();
    }


    void reserve_zero(const int n){
      if(n<=memsize) return;
      //int newsize=n;
      if(dev==0){
	float* newarr=new float[n];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	std::fill(arr+memsize,arr+n,0);
	memsize=n;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, n*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	CUDA_SAFE(cudaMemset(arrg+memsize,0,(n-memsize)*sizeof(float)));
	memsize=n;
      }
      dirg_refresh();
    }


    void dirg_refresh() const{
      if(!dirg) return; 
      delete dirg;
      get_dirg_ptr();
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorPack(const RtensorPack& x):
      dev(x.dev),
      dir(x.dir){
      CNINE_COPY_WARNING();
      tail=x.tail;
      memsize=tail;
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }

    RtensorPack(RtensorPack&& x):
      dev(x.dev),
      dir(std::move(x.dir)){
      CNINE_MOVE_WARNING();
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      dirg=x.dirg; 
    }

    RtensorPack& operator=(const RtensorPack& x)=delete;


  public: // ---- Conversions ---------------------------------------------------------------------------------


    RtensorPack(const rtensor& x){
      CNINE_ASSRT(x.ndims()==2);
      int m=x.dim(1);
      CNINE_CPUONLY();
      dev=x.dev;
      memsize=x.memsize;
      tail=memsize;
      if(x.dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      for(int i=0; i<x.dim(0); i++)
	dir.push_back({i*m,m});
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    RtensorPack(const RtensorPack& x, const int _dev){
      dev=_dev;
      tail=x.tail;
      memsize=x.memsize;
      if(x.dirg) dirg=new IntTensor(*x.dirg);
      if(dev==0){
	arr=new float[memsize];
	if(x.dev==0) std::copy(x.arr,x.arr+tail,arr);
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice)); 
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice)); 
      }
    }


    RtensorPack& to_device(const int _dev){
      if(dev==_dev) return *this;

      if(_dev==0){
	if(dev==1){
	  memsize=tail;
	  delete[] arr;
	  arr=new float[memsize];
	  CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	  CUDA_SAFE(cudaFree(arrg));
	  arrg=nullptr;
	  dev=0;
	}
      }

      if(_dev>0){
	if(dev==0){
	  memsize=tail;
	  if(arrg) CUDA_SAFE(cudaFree(arrg));
	  CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	  CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	  delete[] arr;
	  arr=nullptr;
	  dev=_dev;
	}
      }
      
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }

    int size() const{
      return dir.dim(0);
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


    int addr_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return dir(i,0);
    }

    cnine::Gdims dims_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return dir.row(i,1);
    }

    int dim_of(const int i, const int j) const{
      CNINE_IN_RANGE(i,size());
      return dir(i,1+j);
    }

    float* arr_of(const int i) const{
      if(dev==1) return arrg+addr_of(i);
      return arr+addr_of(i);
    }


    rtensor operator()(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    rtensor view_of_tensor(const int i){
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    const rtensor view_of_tensor(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    //rtensor tensor(const int i) const{
    //assert(i<size());
    //return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    //}

    Rtensor1_view view1_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==2);
      if(dev==1) return Rtensor1_view(arrg+v[0],v[1],1,1);
      return Rtensor1_view(arr+v[0],v[1],1,0);
    }

    Rtensor2_view view2_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==3);
      if(dev==1) return Rtensor2_view(arrg+v[0],v[1],v[2],v[2],1,1);
      return Rtensor2_view(arr+v[0],v[1],v[2],v[2],1,0);
    }

    Rtensor3_view view3_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==4);
      if(dev==1) return Rtensor3_view(arrg+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,1);
      return Rtensor3_view(arr+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,0);
    }


    vector<int> headers(const int i) const{ // legacy
      return dir.row(i);
    }


    IntTensor* get_dirg_ptr(const int _dev=1) const{
      if(!dirg) dirg=new IntTensor(dir,_dev);
      return dirg;
    }


  public: // ---- Push back ----------------------------------------------------------------------------------


    void push_back(const rtensor& x){
      assert(x.dev==dev);
      if(tail+x.asize>memsize)
	reserve(std::max(2*memsize,tail+x.asize));
      if(dev==0){
	std::copy(x.arr,x.arr+x.asize,arr+tail);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+tail,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      dir.push_back(tail,x.dims);
      tail+=x.asize;
    }

    void push_back_raw(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      dir.push_back(tail,_dims);
      tail+=asize;
    }
      
    void push_back_zero(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      if(dev==0){
	std::fill(arr+tail,arr+tail+asize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg+tail,0,asize*sizeof(float)));
      }
      dir.push_back(tail,_dims);
      tail+=asize;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const RtensorPack& x){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr));
      GPUCODE(const float alpha = 1.0; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, tail, &alpha, x.arrg, 1, arrg, 1)));
    }


    void add(const RtensorPack& x, const float c){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr,c));
      GPUCODE(const float alpha = c; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, tail, &alpha, x.arrg, 1, arrg, 1)));
    }


    void add_ReLU(const RtensorPack& x, const float alpha=0.1){
      CNINE_CPUONLY();
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]+=x.arr[i]; else arr[i]+=alpha*x.arr[i]);
      GPUCODE();
    }

    void add_ReLU_back(const RtensorPack& x, const float alpha=0.1){
      CNINE_CPUONLY();
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]=x.arr[i]; else arr[i]=x.arr[i]*alpha);
      GPUCODE();
    }


  public: // ---- Operations ---------------------------------------------------------------------------------

    
    float inp(const RtensorPack& y){
      CNINE_ASSRT(tail==y.tail);
      float t=0;
      CNINE_CPUONLY();
      CPUCODE(for(int i=0; i<tail; i++) t+=arr[i]*y.arr[i];)
      return t;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RtensorPack";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const RtensorPack& v){
      stream<<v.str(); return stream;}

  };


}

#endif
