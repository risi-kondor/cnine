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

#ifndef _auto_array
#define _auto_array

#include "Cnine_base.hpp"


namespace cnine{

  template<typename TYPE>
  class auto_array{
  public:

    mutable TYPE* arr;
    mutable int memsize;
    mutable int _size;

    ~auto_array(){
      if(arr) delete[] arr;
    }

  public: //---- Constructors -------------------------------------


    auto_array(){
      arr=new TYPE[1];
      memsize=1;
      _size=0;
    }

    auto_array(const int n){
      arr=new TYPE[n];
      memsize=n;
      _size=n;
    }


  public: //---- Copying -------------------------------------


    auto_array(const auto_array& x)=delete;


  public: //---- Resizing -------------------------------------


    void resize(const int n){
      if(memsize<n) reserve(n);
      _size=n;
    }

    void reserve(const int x){
      if(x<=memsize) return;
      int new_memsize=std::max(x,2*memsize);
      int* newarr=new TYPE[new_memsize];
      std::copy(arr,arr+memsize,newarr);
      delete[] arr;
      arr=newarr;
      memsize=new_memsize;
    }


  public: //---- Access -------------------------------------


    int size() const{
      return _size();
    }

    TYPE operator[](const int i) const{
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    TYPE& operator[](const int i){
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    int get(const int i) const{
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    void set(const int i, const TYPE& x){
      if(i>=_size) resize(i+1);
      arr[i]=x;
    }


  public: //---- I/O -- -------------------------------------

    
    string str(){
      ostringstream oss;
      oss<<"[";
      for(int i=0; i<size()-1; i++)
	oss<<arr[i]<<",";
      if(_size>0) oss<<arr[_size-1];
      oss<<"]";
      return oss.str();
    }

  };


}

#endif 
