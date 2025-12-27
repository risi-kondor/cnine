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


  template<typename KEY, typename PART>
  class BGtensorPack{
  public:

    using TENSOR=typename PART::TENSOR;
    using BGTENSOR=typename PART::BASE;

    int _nbatch=0;
    Gdims _gdims;
    int _dev=0;
    //DimLabels _labels;
    mutable map<KEY,shared_ptr<PART> > tensors;

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
      CNINE_ASSRT(_nbatch==x._nbatch);
      CNINE_ASSRT(_gdims==x._gdims);
      _dev=x._dev;
      for(auto& p:tensors)
	(*p.second)=*x.tensors[p.first];
      return *this;
    }

    BGtensorPack copy() const{
      BGtensorPack r(_nbatch,_gdims,_dev);
      for(auto& p:tensors)
	r.tensors.emplace(p.first,make_shared<PART>(p.second->copy())); // double copy
      return r;
    }

    void vars_like(const BGtensorPack& x){
      _nbatch=x._nbatch;
      _gdims=x._gdims;
      _dev=x._dev;
    }

    void reset_vars(){
      auto it=tensors.begin();
      if(it==tensors.end()){
	_nbatch=1;
	_gdims=Gdims();
	_dev=0;
      }else{
	_nbatch=it->second->getb();
	_gdims=it->second->gdims();
	_dev=it->second->get_dev();
      }
    }


  public: // ---- ATen --------------------------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each([&](const PART& x){
	  R.push_back(x.torch());});
      return R;
    }

    #endif 


  public: // ---- Autodiff ----------------------------------------------------------------------------------------------------


    BGtensorPack get_grad() const{
      BGtensorPack R;
      R.vars_like(*this);
      for(auto& p:tensors)
	R.tensors.emplace(p.first,make_shared<PART>(p.second->get_grad()));
      return R;
    }

    void set_grad(const BGtensorPack& x) const{
      try{
	for(auto& p:x.tensors)
	  (*this)[p.first].set_grad(new PART(*p.second));
      }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensorPack::set_grad(x): ")+e.what())};
    }

    void add_to_grad(const BGtensorPack& x) const{
      try{
	for(auto& p:x.tensors)
	  (*this)[p.first].add_to_grad(*p.second);
      }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensorPack::add_to_grad(x): ")+e.what())};
    }


  public: // ---- Access ------------------------------------------------------------------------------------------------------


    int get_dev() const{
      return _dev;
    }


  public: // ---- Tensors -------------------------------------------------------------------------------------


    int size() const{
      return tensors.size();
    }

    PART& operator[](const KEY& x) const{
      auto it=tensors.find(x);
      CNINE_ASSRT(it!=tensors.end());
      return *it->second;
    }

    void insert(const KEY& key, const PART& x){
      if(size()==0){
	_nbatch=x.getb();
	_gdims=x.get_gdims();
	_dev=x.get_dev();
      }
      CNINE_ASSRT(x.getb()==getb());
      CNINE_ASSRT(x.gdims()==gdims());
      CNINE_ASSRT(x.get_dev()==get_dev());
      tensors.emplace(key,make_shared<PART>(x));
    }

    void remove(const KEY& key){
      tensors.erase(key);
    }

    void for_each(const std::function<void(const KEY&, const PART&)>& lambda) const{
      for(auto p: tensors)
	lambda(p.first,*p.second);
    }

    //BGtensorPack mapcar(const std::function<PART(const KEY&, const PART& )>& lambda){
    //BGtensorPack r(_nbatch,_gdims,_dev);
      //for(auto p: tensors)
      //r.emplace(p.first,lambda(p.first,p.second));
      //return r;
    //}

    template<typename FUN>
    void zip(const BGtensorPack& y, FUN&& lambda) const{
      for(auto p: tensors){
	lambda(p.first,*p.second,*y.tensors[p.first]);
      }
    }

    template<typename FUN>
    void zip(const BGtensorPack& y, const BGtensorPack& z, FUN&& lambda) const{
      for(auto p: tensors){
	lambda(p.first,*p.second,*y.tensors[p.first],*z.tensors[p.first]);
      }
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
	r.tensors.emplace(p.first,make_shared<PART>(p.second.batch(b)));
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
	r.tensors.emplace(p.first,make_shared<PART>(p.second->cell(ix)));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    BGtensorPack operator+(const BGtensorPack& y) const{
      BGtensorPack R=copy();
      R.add(y);
      return R;
    }

    BGtensorPack operator-(const BGtensorPack& y) const{
      BGtensorPack R=copy();
      R.subtract(y);
      return R;
    }

    template<typename SCALAR, std::enable_if_t<is_numeric_or_complex_v<SCALAR>>* = nullptr>
    BGtensorPack operator*(const SCALAR c) const{
      BGtensorPack R;
      for(auto& p:tensors)
	R.tensors.emplace(p.first,make_shared<PART>((*p.second)*c));
      R.reset_vars();
      return R;
    }

    void add(const BGtensorPack& x) const{
      CNINE_ASSRT(x.size()==size());
      for(auto& p:tensors)
	p.second->add(x[p.first]);
    }

    void subtract(const BGtensorPack& x) const{
      CNINE_ASSRT(x.size()==size());
      for(auto p:x.tensors)
	p.second->subtract(x[p.first]);
    }

    //void add_prod(const BGtensorPack& x, const BGtensor& y) const{
    //for(auto& p: x.tensors)
    //tensors[p.first].add_prod(p.second);
    //}

  public: // ---- Products ------------------------------------------------------------------------------------------------------


    void add_prod(const BGtensorPack& x, const BGTENSOR& y) const{
      zip(x,[&](const BGTENSOR& r, const BGTENSOR& x){
	  r.add_prod(x,y);});
    }

    void add_prod_back0(const BGtensorPack& rg, const BGTENSOR& y) const{
      zip(rg,[&](const KEY& ix, const BGTENSOR& xg, const BGTENSOR& rg){
	  xg.add_prod_XC(rg,y);});
    }

    void add_prod_back1_to(const BGTENSOR& yg, const BGtensorPack& x) const{
      zip(x,[&](const KEY& ix, const BGTENSOR& rg, const BGTENSOR& x){
	  yg.add_prod_XC(rg,x);});
    }


    void add_div(const BGtensorPack& x, const BGTENSOR& y) const{
      zip(x,[&](const BGTENSOR& r, const BGTENSOR& x){
	  r.add_div(x,y);});
    }

    void add_div_back0(const BGtensorPack& rg, const BGTENSOR& y) const{
      zip(rg,[&](const KEY& ix, const BGTENSOR& xg, const BGTENSOR& rg){
	  xg.add_div_back0(rg,y);});
    }

    void add_div_back1_to(const BGTENSOR& yg, const BGtensorPack& x, const BGTENSOR& y) const{
      zip(x,[&](const KEY& ix, const BGTENSOR& rg, const BGTENSOR& x){
	  rg.add_div_back1_to(yg,x,y);});
    }


    void add_mprod(const BGtensorPack& x, const BGTENSOR& y) const{
      zip(x,[&](const BGTENSOR& r, const BGTENSOR& x){
	  r.add_mprod(x,y);});
    }

    void add_mprod_back0(const BGtensorPack& rg, const BGTENSOR& y) const{
      zip(rg,[&](const KEY& ix, const BGTENSOR& xg, const BGTENSOR& rg){
	  xg.add_mprod_back0(rg,y);});
    }

    void add_mprod_back1_to(const BGTENSOR& yg, const BGtensorPack& x) const{
      zip(x,[&](const KEY& ix, const BGTENSOR& rg, const BGTENSOR& x){
	  yg.add_mprod_back1(rg,x);});
    }


  public: // ---- Operations ----------------------------------------------------------------------------------------------------


    BGtensorPack transp(){
      BGtensorPack r(_nbatch,_gdims,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,make_shared<PART>(p.second.transp()));
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
      for_each([&](const KEY& key, const PART& x){
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
