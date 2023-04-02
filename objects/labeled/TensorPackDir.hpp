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


#ifndef _CnineTensorPackDir
#define _CnineTensorPackDir

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "array_pool.hpp"
#include "TensorView.hpp"


namespace cnine{
    

  class TensorPackDir: public array_pool<int>{
  public:


    TensorPackDir(){}

  public: // ---- Constructors -------------------------------------------------------------------------------


    TensorPackDir(const int n, const Gdims& dims):
      TensorPackDir(dims,n){}

    TensorPackDir(const Gdims& dims, const int n):
      array_pool<int>(n,2*dims.size()+1){
      const int m=dims.size();
      const int M=2*m+1;
      Gstrides strides=Gstrides(dims);
      const int t=dims.total();

      for(int i=0; i<n; i++){
	dir.set(i,0,i*M);
	dir.set(i,1,M);
      }

      for(int i=0; i<n; i++){
	for(int j=0; j<m; j++)
	  arr[i*M+j]=dims[j];
 	for(int j=0; j<m; j++)
	  arr[i*M+m+j]=strides[j];
	arr[i*M+2*m]=t*i;
      }
    }

    TensorPackDir(const vector<Gdims>& _dims):
      array_pool<int>(_dims.size()){
      const int n=_dims.size();

      int t=0;
      for(int i=0; i<n; i++)
	t+=2*_dims[i].size()+1;
      reserve(t);

      t=0;
      for(int i=0; i<n; i++){
	dir.set(i,0,tail);
	dir.set(i,1,2*_dims[i].size()+1);
	set_dims(i,_dims[i]);
	set_strides(i,GstridesB(_dims[i]).set_offset(t));
	tail+=2*_dims[i].size()+1;
	t+=_dims[i].total();
      }
    }

    template<typename TYPE>
    TensorPackDir(const initializer_list<TensorView<TYPE> >& list):
      TensorPackDir(convert(list)){}


  private: 


    template<typename TYPE>
    static vector<Gdims> convert(const initializer_list<TensorView<TYPE> >& list){
      vector<Gdims> R;
      for(auto& p:list)
	R.push_back(p.dims);
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return dir.dims[0];
    }

    int ndims(const int i) const{
      CNINE_ASSRT(i<size());
      return (dir(i,1)-1)/2;
    }

    int total() const{
      int t=0;
      for(int i=0; i<size(); i++)
	t+=dims(i).total();
      return t;
    }

    Gdims dims(const int i) const{
      CNINE_ASSRT(i<size());
      return Gdims(vector<int>(arr+dir(i,0),arr+dir(i,0)+ndims(i)));
    }

    GstridesB strides(const int i) const{
      CNINE_ASSRT(i<size());
      const int m=ndims(i);
      return GstridesB(vector<int>(arr+dir(i,0)+m,arr+dir(i,0)+2*m)).set_offset(arr[dir(i,0)+2*m]);
    }

    void set_dims(const int i, const Gdims& x){
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(ndims(i)==x.size());
      std::copy(x.begin(),x.end(),arr+dir(i,0));
    }
      
    void set_strides(const int i, const GstridesB& x){
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(ndims(i)==x.size());
      const int m=ndims(i);
      std::copy(x.begin(),x.end(),arr+dir(i,0)+m);
      arr[dir(i,0)+2*m]=x.offset;
    }

      
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<indent<<"Tensor "<<i<<": "<<dims(i)<<" "<<strides(i)<<endl; 
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorPackDir& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
