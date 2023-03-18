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


#ifndef _CnineMemBlob
#define _CnineMemBlob

#include "Cnine_base.hpp"

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
  class MemBlob{
  public:

    TYPE* arr;
    int dev=0;
    
    ~MemBlob(){
      BLOB_DEBUG("Delete blob.");
      if(dev==0 && arr) {delete[] arr;}
      if(dev==1 && arr) {CUDA_SAFE(cudaFree(arrg));}
    }

  public: // ---- Constructors ------------------------------------------------------------------------------


    MemBlob(const int _memsize, const int _dev=0):
      dev(_dev){
      BLOB_DEBUG("New blob of size "+to_string(_memsize)+".");
      CPUCODE(arr=new TYPE[_memsize];);
      GPUCODE(CUDA_SAFE(cudaMalloc((void **)&arrg, _memsize*sizeof(TYPE))););
    }


  public: // ---- Access ------------------------------------------------------------------------------


    //int device() const{
    //return dev;
    //}


  };

}

#endif 


    /*
    MemBlob(const int _memsize, const fill_zero& dummy, const int _dev=0):
      MemBlob(_memsize,_dev){
      if(dev==0) std::fill(arr,arr+_memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,_memsize*sizeof(TYPE)));
    }

    MemBlob(const int _memsize, const fill_sequential& dummy, const int _dev=0):
      MemBlob(_memsize,_dev){
      CNINE_CPUONLY();
      if(dev==0) {for(int i=0; i<memsize; i++) arr[i]=i;}
    }
    */
