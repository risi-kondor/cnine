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


#ifndef _CnineMemArr
#define _CnineMemArr

#include "Cnine_base.hpp"
#include "MemBlob.hpp"

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
  class MemArr{                                   // An array of size memsize starting at offset offs in blob
  public:

    std::shared_ptr<MemBlob<TYPE> > blob;
    int memsize;
    int offs;

    MemArr(const int _memsize, const int _dev=0):
      blob(new MemBlob<TYPE>(_memsize,_dev)), 
      memsize(_memsize),
      offs(0){}

    MemArr(std::shared_ptr<MemBlob<TYPE> >& _blob, const int _memsize, const int _offs):
      blob(_blob), memsize(_memsize), offs(_offs){}


    // ---- Filled Constructors -----------------------------------------------------------------------------


    MemArr(const int _memsize, const fill_zero& dummy, const int _dev=0):
      MemArr(_memsize,_dev){
      if(device()==0) std::fill(blob->arr,blob->arr+_memsize,0);
      if(device()==1) CUDA_SAFE(cudaMemset(blob->arrarr,0,_memsize*sizeof(TYPE)));
    }

    MemArr(const int _memsize, const fill_sequential& dummy, const int _dev=0):
      MemArr(_memsize,_dev){
      CNINE_ASSRT(device()==0);
      for(int i=0; i<memsize; i++) blob->arr[i]=i;
    }


    // ---- Copying ------------------------------------------------------------------------------------------


    MemArr(const MemArr& x):
      blob(new MemBlob<TYPE>(x.memsize,x.device())),
      memsize(x.memsize),
      offs(0){
      if(device()==0) std::copy(x.blob->arr+x.offs,x.blob->arr+x.offs+memsize,blob->arr);
      if(device()==1) CUDA_SAFE(cudaMemcpy(blob->arr,x.blob->arr+x.offs,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }
      
    MemArr(MemArr&& x):
      blob(x.blob),
      memsize(x.memsize),
      offs(x.offs){}
      
    MemArr& operator=(const MemArr& x){
      CTENS_ASSRT(memsize==x.memsize);
      if(device()==0){
	if(x.device()==0) std::copy(x.get_arr(),x.get_arr()+memsize,get_arr());
	if(x.device()==1) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize*sizeof(TYPE),cudaMemcpyDeviceToHost)); 
      }
      if(device()==1){
	if(x.device()==0) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize*sizeof(float),cudaMemcpyHostToDevice));
	if(x.device()==1) CUDA_SAFE(cudaMemcpy(get_arr(),x.get_arr(),memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }      
    }


    // ---- Access -------------------------------------------------------------------------------------------


    int device() const{
      return blob->dev;
    }

    TYPE* get_arr(){
      return blob->arr+offs;
    } 

    const TYPE* get_arr() const{
      return blob->arr+offs;
    } 

    TYPE operator[](const int i) const{
      return blob->arr[i+offs];
    }

    TYPE& operator[](const int i){
      return blob->arr[i+offs];
    }

    // ---- Operations ---------------------------------------------------------------------------------------


    MemArr offset(const int _offs){
      return MemArr(blob,memsize-_offs,offs+_offs); // TODO
    }

  };

}


    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>

#endif 
