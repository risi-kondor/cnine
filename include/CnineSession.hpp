// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineSession
#define _CnineSession

#include "Cnine_base.hpp"

#ifdef _WITH_CENGINE
#include "CengineSession.hpp"
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif



namespace cnine{

  extern thread_local int nthreads;


  class cnine_session{
  public:

    #ifdef _WITH_CENGINE
    Cengine::CengineSession* cengine_session=nullptr;
    #endif


    cnine_session(const int _nthreads=1){

      nthreads=_nthreads;

      #ifdef _WITH_CENGINE
      cengine_session=new Cengine::CengineSession();
      #endif

      #ifdef _WITH_CUBLAS
      cublasCreate(&cnine_cublas);
      #endif 

    }


    ~cnine_session(){
#ifdef _WITH_CENGINE
      delete cengine_session;
#endif 
    }
    
  };

}


#endif 
