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

    Gstrides(const int i0): vector<int>(1){
      (*this)[0]=i0;
    }

    Gstrides(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    Gstrides(const int i0, const int i1, const int i2): vector<int>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    Gstrides(const int i0, const int i1, const int i2, const int i3): vector<int>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    Gstrides(const int i0, const int i1, const int i2, const int i3, const int i4): vector<int>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }


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

    int back(const int i=0) const{
      return (*this)[size()-1-i];
    }

    bool is_regular() const{
      return regular;
    }

    int offs(const int i0) const{
      return i0*(*this)[0];
    }

    int offs(const int i0, const int i1) const{
      return i0*(*this)[0]+i1*(*this)[1];
    }

    int offs(const int i0, const int i1, const int i2) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2];
    }

    int offs(const int i0, const int i1, const int i2, const int i3) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3];
    }


  public:

    int combine(const vector<int>& v) const{
      int t=0;
      for(auto p:v){
	assert(p<size());
	t+=(*this)[p];
      }
      return t;
    }


  public:

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

  public:

    string str(const string indent="") const{
      ostringstream oss;
      int k=size();
      oss<<indent<<"[";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<"]";
      return oss.str();
    }

  };

}


#endif 
