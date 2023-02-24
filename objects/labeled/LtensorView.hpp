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


#ifndef _CnineLtensorView
#define _CnineLtensorView

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
  class LtensorView{
  public:

    LdimsList ldims;

    ~LtensorView(){
    }


  public:

  public: // ---- Constructors ------------------------------------------------------------------------------


    //LtensorView(const initializer_list<Ldims>& _ldims, const int _dev=0):
    //TensorView<TYPE>(Gdims(convert(_ldims)),_dev),
    //ldims(_ldims){}

  private:

    static vector<vector<int> > convert(const initializer_list<Ldims>& _ldims){
      vector<vector<int> > R;
      for(auto& p:_ldims)
	R.push_back(p);
      return R;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    LtensorView(const LtensorView<TYPE>& x):
      TensorView<TYPE>(x), 
      ldims(x.ldims){}

    LtensorView& operator=(const LtensorView& x){
      assert(ldims==x.ldims);
      TensorView<TYPE>::operator=(x);
    }

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"LtensorView["<<ldims<<"]:";
      
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorView& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
