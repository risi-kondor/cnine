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

#ifndef _CnineTensorSpecBase
#define _CnineTensorSpecBase

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "DimLabels.hpp"


namespace cnine{

  template<typename SPEC>
  class TensorSpecBase{
  public:

    Gdims adims;
    Gdims ddims;
    int nbatch=0;
    int _fcode=0;
    int _dev=0;


    TensorSpecBase(){}

    TensorSpecBase(const Gdims& dims, const DimLabels& labels, const int d){
      if(labels._batched) nbatch=dims[0];
      adims=dims.chunk(labels._batched,labels._narray);
      ddims=dims.chunk(labels._batched+labels._narray);
      _dev=d;
    }


  public: // ---- Construction ------------------------------------------------------------------------------


    SPEC batch(const int b) {nbatch=b; return *this;}
    SPEC batches(const int b) {nbatch=b; return *this;}

    SPEC blocks(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    SPEC blocks(const vector<int>& v) {adims=Gdims(v); return *this;}
    SPEC blocks(const Gdims& v) {adims=v; return *this;}

    SPEC array(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    SPEC array(const vector<int>& v) {adims=Gdims(v); return *this;}
    SPEC array(const Gdims& v) {adims=v; return *this;}

    SPEC dims(const initializer_list<int>& v) {ddims=Gdims(v); return *this;}
    SPEC dims(const vector<int>& v) {ddims=Gdims(v); return *this;}
    SPEC dims(const Gdims& v) {ddims=v; return *this;}
    
    SPEC matrix(const int _n, const int _m) {ddims=Gdims(_n,_m); return *this;}

    SPEC zero() {_fcode=0; return *this;}
    SPEC raw() {_fcode=1; return *this;}
    SPEC ones() {_fcode=2; return *this;}
    SPEC sequential() {_fcode=3; return *this;}
    SPEC gaussian() {_fcode=4; return *this;}

    SPEC dev(const int i) {_dev=i; return *this;}


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      Gdims r;
      if(nbatch>0) r.extend({nbatch});
      if(adims.size()!=0) r.extend(adims);
      if(ddims.size()!=0) r.extend(ddims);
      return r;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=adims.size();
      return r;
    }

    int get_fcode() const{
      return _fcode;
    }

    int get_dev() const{
      return _dev;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"[ ";
      if(nbatch) oss<<"batch="<<nbatch<<" ";
      if(adims.size()>0) oss<<"array="<<nbatch<<" ";
      if(ddims.size()>0) oss<<"dims="<<nbatch<<" ";
      if(_dev>0) oss<<"dev="<<_dev<<" ";
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorSpecBase<SPEC>& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE>
  class Ltensor;

  template<typename TYPE>
  class TensorSpec: public TensorSpecBase<TensorSpec<TYPE> >{
  public:

    typedef TensorSpecBase<TensorSpec<TYPE> > BASE;
    using BASE::BASE;
    TensorSpec(){}
    TensorSpec(const BASE& x): BASE(x){}

    Ltensor<TYPE> operator()(){
      return Ltensor<TYPE>(*this);
    }
    
  };

}

#endif 



/*
  class TensorSpec{
  public:

    Gdims adims;
    Gdims ddims;
    int nbatch=0;
    int _fcode=0;
    int _dev=0;


  public: // ---- Construction ------------------------------------------------------------------------------


    TensorSpec& batch(const int b) {nbatch=b; return *this;}
    TensorSpec& batches(const int b) {nbatch=b; return *this;}

    TensorSpec& blocks(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    TensorSpec& blocks(const vector<int>& v) {adims=Gdims(v); return *this;}
    TensorSpec& blocks(const Gdims& v) {adims=v; return *this;}

    TensorSpec& array(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    TensorSpec& array(const vector<int>& v) {adims=Gdims(v); return *this;}
    TensorSpec& array(const Gdims& v) {adims=v; return *this;}

    TensorSpec& dims(const initializer_list<int>& v) {ddims=Gdims(v); return *this;}
    TensorSpec& dims(const vector<int>& v) {ddims=Gdims(v); return *this;}
    TensorSpec& dims(const Gdims& v) {ddims=v; return *this;}
    
    TensorSpec& matrix(const int _n, const int _m) {ddims=Gdims(_n,_m); return *this;}

    TensorSpec& zero() {_fcode=0; return *this;}
    TensorSpec& raw() {_fcode=1; return *this;}
    TensorSpec& ones() {_fcode=2; return *this;}
    TensorSpec& sequential() {_fcode=3; return *this;}
    TensorSpec& gaussian() {_fcode=4; return *this;}

    TensorSpec& dev(const int i) {_dev=i; return *this;}


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      Gdims r;
      if(nbatch>0) r.extend({nbatch});
      if(adims.size()!=0) r.extend(adims);
      if(ddims.size()!=0) r.extend(ddims);
      return r;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=adims.size();
      return r;
    }

    int get_fcode() const{
      return _fcode;
    }

    int get_dev() const{
      return _dev;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"[ ";
      if(nbatch) oss<<"batch="<<nbatch<<" ";
      if(adims.size()>0) oss<<"array="<<nbatch<<" ";
      if(ddims.size()>0) oss<<"dims="<<nbatch<<" ";
      if(_dev>0) oss<<"dev="<<_dev<<" ";
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorSpec& x){
      stream<<x.str(); return stream;
    }

  };
*/






