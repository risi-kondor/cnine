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


    TensorView(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides):
      arr(_arr),
      dims(_dims), 
      strides(_strides), 
      dev(_arr.device()){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorView(const TensorView<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }
        
    TensorView& operator=(const TensorView& x){
      CNINE_ASSRT(dims==x.dims);
      CNINE_ASSIGN_WARNING();

      if(is_contiguous() && x.is_contiguous()){
	if(device()==0){
	  if(x.device()==0) std::copy(x.get_arr(),x.get_arr()+memsize(),get_arr());
	  if(x.device()==1) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize()*sizeof(TYPE),cudaMemcpyDeviceToHost)); 
	}
	if(device()==1){
	  if(x.device()==0) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize()*sizeof(float),cudaMemcpyHostToDevice));
	  if(x.device()==1) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize()*sizeof(float),cudaMemcpyDeviceToDevice));  
	}      
      }else{
	for_each([&](const Gindex& ix, TYPE& v) {v=x(ix);});
      }

      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int device() const{
      return dev;
    }
    
    int ndims() const{
      return dims.size();
    }

    bool is_regular() const{
      return strides.is_regular(dims);
    }

    bool is_contiguous() const{
      return strides.is_contiguous(dims);
    }

    int asize() const{
      return dims.asize();
    }

    int memsize() const{
      return strides.memsize(dims);
    }

    TYPE* get_arr(){
      return arr.get_arr();
    } 

    const TYPE* get_arr() const{
      return arr.get_arr();
    } 


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


  public: // --- Lambdas ------------------------------------------------------------------------------------


    void for_each(const std::function<void(const Gindex&, TYPE& x)>& lambda){
      dims.for_each_index([&](const Gindex& ix){
	  lambda(ix,arr[strides.offs(ix)]);});
    }

    void for_each(const std::function<void(const Gindex&, TYPE x)>& lambda) const{
      dims.for_each_index([&](const Gindex& ix){
	  lambda(ix,arr[strides.offs(ix)]);});
    }


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

    TensorView<TYPE> slice(const int d, const int i){
      CNINE_CHECK_RANGE(dims.check_in_range_d(d,i,string(__PRETTY_FUNCTION__)));
      return TensorView<TYPE>(arr,dims.remove(d),strides.remove(d).inc_offset(strides[d]*i));
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
	oss<<"[ ";
	for(int i0=0; i0<dims[0]; i0++)
	  oss<<(*this)(i0)<<" ";
	oss<<"]"<<endl;
	return oss.str();
      }

      if(ndims()==2){
	for(int i0=0; i0<dims[0]; i0++){
	  oss<<"[ ";
	  for(int i1=0; i1<dims[1]; i1++)
	    oss<<(*this)(i0,i1)<<" ";
	  oss<<"]"<<endl;
	}
	return oss.str();
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
