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


#ifndef _CnineTensorPackView
#define _CnineTensorPackView

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "GstridesB.hpp"
#include "Gindex.hpp"
#include "MemArr.hpp"
#include "TensorPackDir.hpp"
#include "TensorView.hpp"
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
  class TensorPackView{
  public:

    MemArr<TYPE> arr;
    TensorPackDir dir;
    int dev=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    ///TensorPackView(){
    //cout<<"init2"<<endl;
    //}

    TensorPackView(const TensorPackDir& _dir, const MemArr<TYPE>& _arr):
      arr(_arr),
      dir(_dir),
      dev(_arr.device()){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------

    
    TensorPackView(const TensorPackView& x):
      arr(x.arr),
      dir(x.dir),
      dev(x.dev){}

    TensorPackView(TensorPackView&& x):
      arr(x.arr),
      dir(std::move(x.dir)),
      dev(x.dev){}

    TensorPackView& operator=(const TensorPackView& x){
      CNINE_ASSRT(size()==x.size());
      for(int i=0; i<size(); i++)
	(*this)(i)=x(i);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return dir.size();
    }

    int offset(const int i) const{
      return dir.offset(i);
    }

    Gdims dims(const int i) const{
      return dir.dims(i);
    }

    GstridesB strides(const int i) const{
      return dir.strides(i);
    }

    TensorView<TYPE> operator[](const int i) const{
      return TensorView<TYPE>(arr,dims(i),strides(i));
    }

    TensorView<TYPE> operator()(const int i) const{
      return TensorView<TYPE>(arr,dims(i),strides(i));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorPackView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)(i).str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const TensorPackView& v){
      stream<<v.str(); return stream;}


  };

}

#endif
