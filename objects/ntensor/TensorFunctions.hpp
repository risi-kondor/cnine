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


#ifndef _CnineTensorFunctions
#define _CnineTensorFunctions 

#include "Tensor.hpp"


namespace cnine{


  // ---- Constructors 


  template<typename TYPE>
  inline Tensor<TYPE> Identity(const int n, const int _dev=0){
    return Tensor<TYPE>::identity({n,n},_dev);
  }

  template<typename TYPE>
  inline Tensor<TYPE> UnitVec(const int n, const int i, const int _dev=0){
    Tensor<TYPE> R({n},fill_zero(),_dev);
    R.set(i,1);
    return R;
  }




  template<typename TYPE>
  inline Tensor<TYPE> Tensr(const initializer_list<TYPE>& list, const int _dev=0){
    Tensor<TYPE> T(Gdims(list.size()),_dev); 
    int i=0;
    for(auto& p: list)
      T.set(i++,p);
    return T;
  }

  template<typename TYPE>
  inline Tensor<TYPE> Tensr(const initializer_list<initializer_list<TYPE> >& list, const int _dev=0){
    int n0=list.size();
    CNINE_ASSRT(n0>0);
    int n1=list.begin()->size();
    Tensor<TYPE> T(Gdims(n0,n1),_dev); 
    int i=0;
    for(auto& p: list){
      int j=0;
      for(auto& q: p)
	T.set(i,j++,q);
      i++;
    }
    return T;
  }


  // ---- Elementwise operations 


  template<typename TYPE>
  inline Tensor<TYPE> operator+(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R(x);
    R.add(y);
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator-(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R(x);
    R.subtract(y);
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator*(const TYPE c, const TensorView<TYPE>& x){
    Tensor<TYPE> R=Tensor<TYPE>::zeros_like(x);
    R.add(x,c);
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> prod(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R=Tensor<TYPE>::zero(x.dims,x.dev);
    R.add_prod(x,y);
    return R;
  }


  // ---- Matrix products


  template<typename TYPE>
  inline Tensor<TYPE> operator*(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R=Tensor<TYPE>::zero(x.dims.Mprod(y.dims),x.dev);
    if(R.asize()>0) R.add_mprod(x,y);
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator*(const TensorView<TYPE>& x, const Transpose<TensorView<TYPE> >& y){
    return x*(y.obj.transp());
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator*(const TensorView<TYPE>& x, const Transpose<Tensor<TYPE> >& y){
    return x*(y.obj.transp());
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator*(const Transpose<TensorView<TYPE> >& x, const TensorView<TYPE>& y){
    return (x.obj.transp())*y;
  }

  template<typename TYPE>
  inline Tensor<TYPE> operator*(const Transpose<Tensor<TYPE> >& x, const TensorView<TYPE>& y){
    return (x.obj.transp())*y;
  }


  // ---- Conjugation


  template<typename TYPE>
  inline Tensor<TYPE> conjugate(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return y*x*y.transp();
  }


  // ---- Concatenation 


  template<typename TYPE>
  inline Tensor<TYPE> cat(const int d, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    CNINE_ASSRT(x.dims.size()==y.dims.size());
    CNINE_ASSRT(d<x.dims.size());
    Gdims dims=x.dims;
    dims[d]+=y.dims[d];
    Tensor<TYPE> R(dims);
    R.block(x.dims)=x;
    Gindex ix(x.dims.size(),fill_zero());
    ix[d]=x.dims[d];
    R.block(y.dims,ix)=y;
    return R;
  }

  template<typename VEC, typename OBJ, typename TYPE>
  inline Tensor<TYPE> cat(const int d, const VEC& vec, std::function<TensorView<TYPE>(const OBJ& x) > lambda){
    int n=0;
    Gdims dims;
    for(auto& p:vec){
      auto x=lambda(p);
      if(n==0) dims=x.dims;
      CNINE_ASSRT(d<x.dims.size());
      n+=x.dims[d];
    }
    dims[d]=n;

    int offs=0;
    Tensor<TYPE> R(dims,fill_raw());
    for(auto& p:vec){
      auto x=lambda(p);
      Gindex ix(x.dims.size(),fill_zero());
      ix[d]=offs;
      R.block(x.dims,ix)=x;
      offs+=x.dims[d];
    }      
    return R;
  }


  // ---- Direct sums 


  template<typename TYPE>
  inline Tensor<TYPE> oplus(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    CNINE_ASSRT(x.ndims()==y.ndims());
    Tensor<TYPE> R=Tensor<TYPE>::zero(x.dims+y.dims,x.dev);
    R.block(x.dims)=x;
    R.block(y.dims,Gindex(x.dims))=y;
    return R;
  }
  

  // ---- Tensor Products 


  template<typename TYPE>
  inline Tensor<TYPE> tprod(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    Tensor<TYPE> R=Tensor<TYPE>::zero(tprod(x.dims,y.dims),x.dev);
    R.add_tprod(x,y);
    return R;
  }


  // ---- Other Operations 
  

  template<typename TYPE>
  inline Tensor<TYPE> diag(const TensorView<TYPE>& x){
    CNINE_ASSRT(x.ndims()==1);
    Tensor<TYPE> R=Tensor<TYPE>::zero({x.dims[0],x.dims[0]},x.dev);
    R.diag()=x;
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> broadcast(const int d, const int n, const TensorView<TYPE>& x){
    CNINE_ASSRT(d<x.dims.size()+1);
    Gdims dims(x.dims);
    dims.insert(d,n);
    Tensor<TYPE> R=Tensor<TYPE>::raw(dims,x.dev);
    for(int i=0; i<n; i++)
      R.slice(d,i)=x;
    return R;
  }

  template<typename TYPE>
  inline Tensor<TYPE> sum(const int d, const TensorView<TYPE>& x){
    auto dims=x.dims.remove(d);
    Tensor<TYPE> R(dims,fill_zero());
    R.add_sum(d,x);
    return R;
  }

}

#endif 
