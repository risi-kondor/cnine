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


#ifndef _CnineTensor
#define _CnineTensor

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "BatchedTensorView.hpp"

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
  class Tensor: public TensorView<TYPE>{
  public:

    using TensorView<TYPE>::TensorView;
    using TensorView<TYPE>::arr;
    using TensorView<TYPE>::dims;
    using TensorView<TYPE>::strides;
    using TensorView<TYPE>::dev;

    //using TensorView<TYPE>::operator=;
    using TensorView<TYPE>::ndims;
    using TensorView<TYPE>::dim;
    using TensorView<TYPE>::set;
    using TensorView<TYPE>::row;
    using TensorView<TYPE>::transp;
    using TensorView<TYPE>::fuse01;
    using TensorView<TYPE>::split0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Tensor():
      TensorView<TYPE>(MemArr<TYPE>(1),{1},{1}){}

    Tensor(const Gdims& _dims, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=dummy.v;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_identity& dummy, const int _dev=0):
      Tensor(_dims,fill_zero()){
      CNINE_ASSRT(ndims()==2);
      CNINE_ASSRT(dim(0)==dim(1));
      int N=dim(0);
      for(int i=0; i<N; i++)
	set(i,i,1.0);
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_random_unitary& dummy, const int _dev=0):
      Tensor(_dims,fill_zero(),0){
      CNINE_ASSRT(ndims()==2);
      CNINE_ASSRT(dim(0)==dim(1));
      int N=dim(0);
      for(int i=0; i<N; i++){
	auto v=Tensor({N},fill_gaussian(),0);
	for(int j=0; j<i; j++){
	  auto u=row(j); 
	  v=v-inp(u,v)*u;
	}
	row(i).add(v,TYPE(1.0)/norm(v));
      }
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Tensor<TYPE> zero(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> ones(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_constant<TYPE>(1),_dev);
    }

    static Tensor<TYPE> constant(const Gdims& _dims, const TYPE v, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_constant<TYPE>(v),_dev);
    }

    static Tensor<TYPE> identity(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_identity(),_dev);
    }

    static Tensor<TYPE> random_unitary(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_random_unitary(),_dev);
    }

    static Tensor<TYPE> sequential(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_sequential(),_dev);
    }

    static Tensor<TYPE> randn(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_gaussian(),_dev);
    }

    static Tensor<TYPE> gaussian(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_gaussian(),_dev);
    }

    static Tensor<TYPE> zeros_like(const TensorView<TYPE>& x, const int _dev=-1){
      if(_dev==-1) return Tensor<TYPE>(x.dims,fill_zero(),x.dev);
      else return Tensor<TYPE>(x.dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> identity_like(const TensorView<TYPE>& x, const int _dev=-1){
      if(_dev==-1) return Tensor<TYPE>(x.dims,fill_identity(),x.dev);
      else return Tensor<TYPE>(x.dims,fill_identity(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    Tensor(const Tensor<TYPE>& x):
      Tensor(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    Tensor(const Tensor<TYPE>& x, const nowarn_flag& dummy):
      Tensor(x.dims,x.dev){
      view()=x.view();
    }
        
    Tensor(const Tensor<TYPE>&& x):
      TensorView<TYPE>(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    Tensor& operator=(const Tensor& x){
      CNINE_ASSIGN_WARNING();
      arr=MemArr<TYPE>(x.dims.total(),x.dev);
      dims=x.dims;
      strides=GstridesB(dims);
      dev=x.dev;
      TensorView<TYPE>::operator=(x);
      return *this;
    }
    
    Tensor& operator=(Tensor&& x){
      CNINE_MOVEASSIGN_WARNING();
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
      arr=x.arr;
      return *this;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    // Doesn't work 
    //operator BatchedTensorView<TYPE>() const{
    //return TensorView(*this);
    //}


  public: // ---- Transport -----------------------------------------------------------------------------------


    Tensor(const TensorView<TYPE>& x, const int _dev):
      Tensor(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<Tensor&>(*this)=Tensor(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    Tensor(const at::Tensor& T):
      Tensor(Gdims(T),T.type().is_cuda()){
      (*this)=T;
    }

    #endif


  public: // ---- Eigen --------------------------------------------------------------------------------------


#ifdef _WITH_EIGEN

    Tensor(const Eigen::VectorXf& x):
      Tensor(Gdims(x.size())){
      int n=dims[0];
      for(int i=0; i<n; i++) 
	  set(i,x(i));
    }

    Tensor(const Eigen::VectorXd& x):
      Tensor(Gdims(x.size())){
      int n=dims[0];
      for(int i=0; i<n; i++) 
	  set(i,x(i));
    }

    Tensor(const Eigen::MatrixXf& x):
      Tensor(Gdims(x.rows(),x.cols())){
      int n=dims[0];
      int m=dims[1];
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  set(i,j,x(i,j));
    }

    Tensor(const Eigen::MatrixXd& x):
      Tensor(Gdims(x.rows(),x.cols())){
      int n=dims[0];
      int m=dims[1];
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  set(i,j,x(i,j));
    }

#endif 


  public: // ---- Views -------------------------------------------------------------------------------------


    Tensor(const TensorView<TYPE>& x):
      Tensor(x.dims,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    TensorView<TYPE> view(){
      return TensorView<TYPE>(*this);
    }

    const TensorView<TYPE> view() const{
      return TensorView<TYPE>(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    Tensor operator*(const TensorView<TYPE>& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	Tensor R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	Tensor R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	Tensor R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return Tensor();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Tensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"Tensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Tensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  // ---- Functions ----------------------------------------------------------------------------------------------


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
    R.add_mprod(x,y);
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

}

#endif
