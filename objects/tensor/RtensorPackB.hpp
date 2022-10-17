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


#ifndef _RtensorPackB
#define _RtensorPackB

#include "RtensorPack.hpp"


namespace cnine{


  class RtensorPackB: public RtensorPack{
  public:

    typedef RtensorA rtensor;
    int nc;


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPackB(){}

    RtensorPackB(const int ndims, const int _nc, const int _dev):
      RtensorPack(ndims,_dev), nc(_nc){}

    RtensorPackB(const IntTensor& _dir, const int _nc, const int _dev):
      RtensorPack(_dir,_dev), nc(_nc){}

    template<typename FILLTYPE>
    RtensorPackB(const int _N, const Gdims& _dims, const FILLTYPE& dummy, const int _dev=0):
      RtensorPack(_N,_dims,dummy,_dev), nc(_dims.back()){}

    template<typename FILLTYPE>
    RtensorPackB(const cnine::array_pool<int>& dims, const FILLTYPE& dummy, const int _dev=0):
      RtensorPack(dims,dummy,_dev){
      CNINE_ASSRT(dims.size()>0);
      nc=dims(0).back();
    }

    /*
    RtensorPackB(const int _N, const Gdims& _dims, const cnine::fill_raw& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPackB(const int _N, const Gdims& _dims, const cnine::fill_zero& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1){CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)))};
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPackB(const int _N, const Gdims& _dims, const cnine::fill_gaussian& dummy, const int _dev=0):
      RtensorPack(_dims.size(),0){
      int asize=_dims.asize();
      reserve(_N*asize);
      normal_distribution<double> distr;
      for(int i=0; i<_N*asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<_N; i++){
	dir.push_back(i*asize,_dims);
      }
      tail=_N*asize;
      to_device(_dev);
    }

    RtensorPackB(const cnine::array_pool<int>& dimensions, const cnine::fill_zero& dummy, const int _dev=0){
      dev=_dev;
      dir=IntTensor(Gdims(0,dimensions(0).size()+1),cnine::fill_noalloc());

      int reserve_size=0;
      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	reserve_size+=t;
      }
      reserve(reserve_size);
      if(dev==0) std::fill(arr,arr+reserve_size,0);
      if(dev==1){CUDA_SAFE(cudaMemset(arrg,0,reserve_size*sizeof(float)))};

      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	dir.push_back(tail,v);
	tail+=t;
      }
    }
    */

  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorPackB zeros_like(const RtensorPackB& x){
      RtensorPackB R(x.dir,x.nc,x.dev);
      R.reserve(x.tail);
      if(x.dev==0) std::fill(R.arr,R.arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R.arrg,0,R.tail*sizeof(float)));
      R.tail=x.tail;
      return R;
    }

    static RtensorPackB* new_zeros_like(const RtensorPackB& x){
      RtensorPackB*  R=new RtensorPackB(x.dir,x.nc,x.dev);
      R->reserve(x.tail);
      if(x.dev==0) std::fill(R->arr,R->arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R->arrg,0,R->tail*sizeof(float)));
      R->tail=x.tail;
      return R;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorPackB(const RtensorPackB& x):
      RtensorPack(x), nc(x.nc){}

    RtensorPackB(RtensorPackB&& x):
      RtensorPack(std::move(x)), nc(x.nc){}

    RtensorPackB& operator=(const RtensorPackB& x){
      RtensorPack::operator=(x);
      nc=x.nc;
      return *this;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    RtensorPackB(const rtensor& x):
      RtensorPack(x){
      nc=x.get_dim(1);
    }

    RtensorPackB(rtensor&& x):
      RtensorPack(x){
      nc=x.get_dim(1);
    }

    rtensor tensor() const{
      return rtensor({size(),nc},get_arr(),dev);
    }

    rtensor view_as_matrix() const{
      return rtensor::view_of_blob({tail/nc,nc},get_arr(),dev);
    }

    Rtensor2_view matrix_view() const{
      return Rtensor2_view(get_arr(),tail/nc,nc,nc,1,dev);
    }

    #ifdef _WITH_ATEN
    RtensorPackB(const at::Tensor& T):
      RtensorPackB(rtensor(T)){
      assert(size()>0);
      nc=dim_of(0,0);
    }
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    RtensorPackB(const RtensorPackB& x, const int _dev): 
      RtensorPack(x,_dev){
      nc=x.nc;
    }

    RtensorPackB& to_device(const int _dev){
      RtensorPack::to_device(_dev);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const RtensorPackB& x, const rtensor& y){
      PTENS_ASSRT(x.size()==size());
      view_as_matrix().add_mprod(x.view_as_matrix(),y);
      //matrix_view().add_mprod(x.matrix_view(),y.view2());
    }

    void add_mprod_back0(const RtensorPackB& g, const rtensor& y){
      PTENS_CPUONLY();
      view_as_matrix().add_Mprod_AT(g.view_as_matrix(),y);
    }

    void add_mprod_back1_to(rtensor& r, const RtensorPackB& x) const{
      PTENS_CPUONLY();
      r.add_Mprod_TA(x.view_as_matrix(),view_as_matrix());
    }

    void add_bias(const rtensor& b){
      matrix_view().add_broadcast0(b.view1());
    }

    void add_bias_back1_to(const rtensor& b){
      matrix_view().sum0_into(b.view1());
    }

    void add_linear(const RtensorPackB& x, const rtensor& y, const rtensor& b){
      add_mprod(x,y);
      add_bias(b);
    }

    void add_linear_back0(const RtensorPackB& g, const rtensor& y){
      add_mprod_back0(g,y);
    }

    void add_linear_back1_to(rtensor& r, const RtensorPackB& x) const{
      add_mprod_back1_to(r,x);
    }

    void add_linear_back2_to(rtensor& b){
      add_bias_back1_to(b);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RtensorPackB";
    }

    friend ostream& operator<<(ostream& stream, const RtensorPackB& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
