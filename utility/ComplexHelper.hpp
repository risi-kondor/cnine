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

#ifndef _ComplexHelper
#define _ComplexHelper

#include "Cnine_base.hpp"

namespace cnine{


template<typename T>
struct ComplexHelper{
  ComplexHelper(const bool v=false){
    if(v) cout<<"error!"<<endl;}
  bool operator()()const{return false;}
  bool flip(){return false;}
};

template<typename U>
struct ComplexHelper<std::complex<U>> {
  bool value=false;
  ComplexHelper(const bool v=false):value(v){}
  bool operator()()const{return value;}
  bool flip(){value=!value; return value;}
};


template<typename TENSOR>
TENSOR herm(const TENSOR& x){
  CNINE_ASSRT(x.ndims()>=2);
  TENSOR r(x);
  r.dims=r.dims.transp();
  r.strides=r.strides.transp();
  r.is_conj.flip();
  return r;
}

}


#endif 
