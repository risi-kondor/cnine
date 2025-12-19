/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _cnineBGtensorPack
#define _cnineBGtensorPack

#include "BGtensor.hpp"
//#include "BGtensorApackSpec.hpp"


namespace cnine{


  template<typename KEY, typename TENSOR>
  class BGtensorPack{
  public:

    int _nbatch=0;
    Gdims _gdims;
    int _dev=0;
    //DimLabels _labels;
    mutable map<KEY,TENSOR> tensors;

    BGtensorPack(){}

    ~BGtensorPack(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    BGtensorPack(const int __nbatch, const Gdims& __gdims,  const int __dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      _dev(__dev){}

    //BGtensorPack(const int __nbatch, const Gdims& __gdims,  const DimLabels& __labels, const int __dev=0):
      //_nbatch(__nbatch),
      //_gdims(__gdims),
      //_dev(__dev),
      //_labels(__labels){}

    
  public: // ---- Copying -----------------------------------------------------------------------------------


    BGtensorPack(const BGtensorPack& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      _dev(x._dev){
      for(auto& p:x.tensors)
	tensors.emplace(p.first,p.second);
    }
    
    BGtensorPack(BGtensorPack&& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      _dev(x._dev),
      tensors(std::move(x.tensors)){
    }
      
    BGtensorPack& operator=(const BGtensorPack& x){
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      _dev=x._dev;
      for(auto& p:tensors)
	p.second=x.tensors[p.first];
      return *this;
    }

    BGtensorPack copy() const{
      BGtensorPack r(_nbatch,_gdims,_dev);
      for(auto& p:tensors)
	r.tensors[p.first]=p.second->copy();
      return r;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each([&](const TENSOR& x){
	  R.push_back(x.torch());});
      return R;
    }

    #endif 


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_dev() const{
      return _dev;
    }


  public: // ---- Tensors -------------------------------------------------------------------------------------


    int size() const{
      return tensors.size();
    }

    TENSOR operator[](const KEY& x) const{
      CNINE_ASSRT(tensors.find(x)!=tensors.end());
      return const_cast<BGtensorPack&>(*this).tensors[x];
    }

    void insert(const KEY& key, const TENSOR& x){
      if(size()==0){
	_nbatch=x.getb();
	_gdims=x.get_gdims();
	_dev=x.get_dev();
      }
      CNINE_ASSRT(x.getb()==getb());
      CNINE_ASSRT(x.gdims()==gdims());
      CNINE_ASSRT(x.get_dev()==get_dev());
      tensors.emplace(key,x);
    }

    void for_each(const std::function<void(const KEY&, const TENSOR&)>& lambda) const{
      for(auto p: tensors)
	lambda(p.first,p.second);
    }

    BGtensorPack mapcar(const std::function<TENSOR(const KEY&, const TENSOR& )>& lambda){
      BGtensorPack r(_nbatch,_gdims,_dev);
      //for(auto p: tensors)
      //r.emplace(p.first,lambda(p.first,p.second));
      return r;
    }



  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>1;
    }

    int nbatch() const{
      return _nbatch;
    }

    int getb() const{
      return _nbatch;
    }

    BGtensorPack batch(const int b) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b>=0 && b<_nbatch);
      BGtensorPack r(0,_gdims,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.batch(b));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const BGtensorPack& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return _gdims.size()>0;
    }

    bool has_grid() const{
      return _gdims.size()>0;
    }

    int ngdims() const{
      return _gdims.size();
    }

    Gdims gdims() const{
      return _gdims;
    }

    Gdims get_gdims() const{
      return _gdims;
    }

    BGtensorPack cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      BGtensorPack r(_nbatch,cnine::Gdims(),_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.cell(ix));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const BGtensorPack& x){
      CNINE_ASSRT(x.size()==size());
      for(auto p:x.tensors)
	tensors[p.first].add(p.first);
    }

    void subtract(const BGtensorPack& x){
      CNINE_ASSRT(x.size()==size());
      for(auto p:x.tensors)
	tensors[p.first].subtract(p.first);
    }

    //void add_prod(const BGtensorPack& x, const BGtensor& y) const{
    //for(auto& p: x.tensors)
    //tensors[p.first].add_prod(p.second);
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------


    BGtensorPack transp(){
      BGtensorPack r(_nbatch,_gdims,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.transp());
      return r;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BGtensorPack";
    }

    string repr() const{
      ostringstream oss;
      oss<<"BGtensorPack(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      if(_dev>0) oss<<"dev="<<_dev<<",";
      if(is_batched()||is_grid()||_dev>0) oss<<"\b";
      oss<<")";
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      for_each([&](const KEY& key, const TENSOR& x){
	  oss<<indent<<"Tensor "<<key<<":"<<endl;
	  oss<<x.str(indent)<<endl;
	});
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BGtensorPack& x){
      stream<<x.str(); return stream;
    }
  };

}

#endif 
