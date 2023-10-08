/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _int_pool
#define _int_pool

#include "Cnine_base.hpp"


namespace cnine{

  class int_pool{
  public:

    int n;
    int last=-1;
    int memsize;
    int* arr;

    ~int_pool(){
      delete[] arr;
    }

    int_pool(const int _n, const int _m):
      n(_n){
      memsize=_n+_m+2;
      arr=new int[memsize];
      arr[0]=n;
      arr[1]=n+2; 
    }
    

  public: // ---- Copying --------------------------------------------------


    int_pool(const int_pool& x):
      n(x.n), last(x.last), memsize(x.memsize){
      arr=new int[memsize];
      std::copy(x.arr,x.arr+memsize,memsize);
    }

    int_pool(int_pool&& x):
      n(x.n), last(x.last), memsize(x.memsize){
      arr=x.arr;
      x.arr=nullptr;
    }

    int_pool operator=(const int_pool& x)=delete;


  public: // ---- Access ---------------------------------------------------


    int tail() const{
      return arr[last+2];
    }

    int addr_of(const int i) const{
      CNINE_ASSRT(i<n);
      return arr[i+1];
    }

    int size_of(const int i) const{
      CNINE_ASSRT(i<n);
      return arr[i+2]-arr[i+1];
    }

    int operator()(const int i, const int j) const{
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<size_of(i));
      return arr[arr[i+1]+j];
    }

    int& operator()(const int i, const int j){
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<size_of(i));
      return arr[arr[i+1]+j];
    }

    int add_vec(const int m){
      arr[last+2]=arr[last+1]+m;
      last++;
      return arr[last+1];
    }

  };

};

#endif 

