//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensorB_array
#define _CnineCtensorB_array

#include "CtensorB.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  class CtensorB_array: public CtensorB{
  public:

    int ak;

    CtensorB_array(){}

    ~CtensorB_array(){
    }

    string classname() const{
      return "CtensorB_array";
    }

    string describe() const{
      return "CtensorB_array"+dims.str();
    }


  public: // ---- Constructors -----------------------------------------------------------------------------


    CtensorB_array(const Gdims& _adims, const Gdims& _dims, const int _dev=0): 
      CtensorB(Gdims(_adims,_dims),_dev), ak(_adims.size()){}


  public: // ---- Filled constructors -----------------------------------------------------------------------


    template<typename FILLTYPE>
    CtensorB_array(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& dummy, const int _dev=0): 
      CtensorB(Gdims(_adims,_dims),dummy,_dev), ak(_adims.size()){}

    
  public: // ---- Named constructors -------------------------------------------------------------------------


    static CtensorB_array raw(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_raw(),_dev);
    }

    static CtensorB_array zero(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_zero(),_dev);
    }

    static CtensorB_array zeros(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_zero(),_dev);
    }

    static CtensorB_array ones(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_ones(),_dev);
    }

    static CtensorB_array sequential(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_sequential(),_dev);
    }

    static CtensorB_array gaussian(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorB_array(_adims,_dims,fill_gaussian(),_dev);
    }

    //static CtensorB_array zeros_like(const CtensorB_array& x){
    //return CtensorB_array(x.dims,fill_zero(),x.dev);
    //}


  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorB_array(const CtensorB_array& x):
      CtensorB(x), ak(x.ak){}

    CtensorB_array(CtensorB_array&& x):
      CtensorB(std::move(x)), ak(x.ak){}

    CtensorB_array& operator=(const CtensorB_array& x){
      CtensorB::operator=(x);
      ak=x.ak;
      return *this;
    }

    CtensorB_array& operator=(CtensorB_array&& x){
      CtensorB::operator=(std::move(x));
      ak=x.ak;
      return *this;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    CtensorB_array(const CtensorB& x, const int _ak):
      CtensorB(x){
      if(_ak<0) ak=dims.size()+_ak;
      else ak=_ak;
    }

    CtensorB_array(CtensorB&& x, const int _ak):
      CtensorB(std::move(x)){
      if(_ak<0) ak=dims.size()+_ak;
      else ak=_ak;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    CtensorB_array(const at::Tensor& T, const int _ak):
      CtensorB_array(CtensorB(T),_ak){}

    static CtensorB_array view(at::Tensor& T, const int _ak){
      return CtensorB_array(CtensorB::view(T),_ak);
    }

    static CtensorB_array* viewp(at::Tensor& T, const int _ak){
      return new CtensorB_array(CtensorB::view(T),_ak);
    }

#endif 


  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims get_adims() const{
      return dims.chunk(0,ak);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorB_array gather(const Rmask1& mask){
      Gdims _dims(dims);
      _dims[0]=mask.N0;
      CtensorB_array R(CtensorB::zero(_dims,dev),ak);
      R.add_gather(*this,mask);
      return R;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_gather(const CtensorB& x, const Rmask1& mask){
      if(ak!=1) throw std::invalid_argument("CtensorB_array::add_gather(const CtensorB&, const Rmask1&): number of array arguments must be 1.");
      assert(x.dims.size()==dims.size());
      Aggregator(viewx(),x.viewx(),mask);
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      assert(ak<dims.size());
      Gdims arraydims=dims.chunk(0,ak);
      arraydims.foreach_index([&](const vector<int>& ix){
	  oss<<indent<<"Cell"<<Gindex(ix)<<endl;
	  oss<<get_cell(ix).str(indent)<<endl;
	});
      return oss.str();
    }

    string repr() const{
      return "<cnine::CtensorB_array"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorB_array& x){
      stream<<x.str(); return stream;}
   




    

  };

}

#endif 
