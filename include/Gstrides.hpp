// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef __Gstrides
#define __Gstrides

#include "Cnine_base.hpp"
#include "Gdims.hpp"

namespace cnine{


  class Gstrides: public vector<int>{
  public:

    bool regular;

    Gstrides(){}

    Gstrides(const int k, const fill_raw& dummy): 
      vector<int>(k){}

    Gstrides(const initializer_list<int>& lst):
      vector<int>(lst){}

    Gstrides(const Gdims& dims, const int s0=1): 
      vector<int>(dims.size()){
      int k=dims.size();
      assert(k>0);
      (*this)[k-1]=s0;
      for(int i=k-2; i>=0; i--)
	(*this)[i]=(*this)[i+1]*dims[i+1];
      regular=true;
    }

    Gstrides(const vector<int>& x):
      vector<int>(x){
    }


  public:

    int operator()(const int i) const{
      return (*this)[i];
    }

    bool is_regular() const{
      return regular;
    }

    Gstrides chunk(const int beg, int n=-1) const{
      if(n==-1) n=size()-beg;
      Gstrides R(n);
      for(int i=0; i<n; i++)
	R[i]=(*this)[beg+i];
      return R;
    }

    int offs(int j, const Gstrides& source) const{
      assert(source.size()==size());
      int t=0;
      for(int i=0; i<size(); i++){
	int r=j/source[i];
	t+=(*this)[i]*r;
	j-=r*source[i];
      }
      return t;
    }

  };

}


#endif 
