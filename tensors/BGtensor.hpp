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

#ifndef _BGtensor
#define _BGtensor

#include "TensorView.hpp"
#include "MultiLoop.hpp"

#include "ForEachCellMulti.hpp"
#include "ForEachCellMultiScalar.hpp"
#include "BGtensor_reconcilers.hpp"


namespace cnine{

  template<typename TYPE>
  class BGtensor: public TensorView<TYPE>{
  public:
  
    using TENSOR=TensorView<TYPE>;

    using TENSOR::arr;
    using TENSOR::dims;
    using TENSOR::strides;
    using TENSOR::dev;

    using TENSOR::ndims;
    using TENSOR::get_dev;
    using TENSOR::slice;
    using TENSOR::slices;
    using TENSOR::repr;

    int ng=0;


  public: // ---- Constructors -------------------------------------------------------------------------------


    BGtensor(){}

    BGtensor(const int b, const Gdims gdims, const Gdims cdims, const int fill=0, const int _dev=0):
      TENSOR(Gdims::cat(b,gdims,cdims),fill,_dev){
      ng=gdims.size();
    }

    BGtensor(const initializer_list<BGtensor>& v):
      BGtensor(vector<BGtensor>(v)){}

    BGtensor(const vector<BGtensor>& v):
      TENSOR([](const vector<BGtensor>& v){
	  CNINE_ASSRT(v.size()>0); 
	  auto& x=*v.begin(); 
	  return TENSOR(Gdims::cat(x.getb()*v.size(),x.gdims(),x.cdims()),0,x.get_dev());}){
      auto& x=*v.begin(); 
      int _b=x.getb();
      Gdims _gdims=x.gdims();
      Gdims _cdims=x.cdims();
      int offs=0;
      for(auto& p:v){
	CNINE_ASSRT(p.getb()==_b);
	CNINE_ASSRT(p.gdims()==_gdims);
	CNINE_ASSRT(p.cdims()==_cdims);
	slices(0,offs,_b).add(p);
	offs+=_b;
      }
    }

    //BGtensor(const MemArr<TYPE>& _arr, const int _ng, Gdims& _dims, const GstridesB& _strides):
    //TENSOR(_arr,_dims,_strides){
    //ng=_ng;
    //}

    BGtensor(const MemArr<TYPE>& _arr, const int _b, const Gdims& _gdims, const Gdims& _cdims,
      const int _bstride, const GstridesB& _gstrides, const GstridesB& _cstrides):
      TENSOR(_arr,Gdims(_b,_gdims,_cdims),GstridesB(_bstride,_gstrides,_cstrides)){
      CNINE_ASSRT(_gdims.size()==_gstrides.size());
      CNINE_ASSRT(_cdims.size()==_cstrides.size());
      ng=_gdims.size();
    }

    BGtensor(const int _ng, const TENSOR& x):
      TENSOR(x), ng(_ng){
      CNINE_ASSRT(_ng<=ndims()-1);
    }

    BGtensor(const TENSOR& x, const bool _batched=0, const int _ng=0):
      TENSOR(x), ng(_ng){
      if(!_batched){
	dims=dims.insert(0,1);
	strides=strides.insert(0,0);
      }
    }

    BGtensor like(const TENSOR& x) const{
      return BGtensor(ng,x);
    }

    BGtensor zeros_like() const{
      return BGtensor(ng,TENSOR::zeros_like());
    }

    void reset(const BGtensor& x){
      arr=x.arr;
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
      ng=x.ng;
    }

    void reset(const int b, const Gdims gdims, const Gdims cdims, const int fill=0, const int _dev=0){
      BGtensor r(b,gdims,cdims,fill,_dev);
      arr=r.arr;
      dims=r.dims;
      strides=r.strides;
      dev=r.dev;
      ng=r.ng;
    }


  public: // ---- Copying ---------- -------------------------------------------------------------------------

    
    BGtensor(const BGtensor& x):
      TENSOR(x),
      ng(x.ng){}
    
    BGtensor& operator=(const BGtensor& x) const{
      CNINE_ASSRT(ng==x.ng);
      TENSOR::operator=(x);
      return *this;
    }
    
    BGtensor copy() const{
      return BGtensor(ng,TENSOR::copy());
    }
      
    BGtensor conjugate(){
      BGtensor R(*this);
      R.is_conj.flip();
      return R;
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static BGtensor scalar(const TENSOR& x){
      CNINE_ASSRT(x.ndims()==1);
      return BGtensor(x,1);
    }

    static BGtensor scalar(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),true);
    }


    static BGtensor vectr(const TENSOR& x){
      return BGtensor(x,x.ndims()>1,std::max(0,x.ndims()-2));
    }


    static BGtensor tensor(const TENSOR& x){
      return BGtensor(x,0);
    }

    //static BGtensor tensor(const initializer_list<TYPE>& v){
    //return BGtensor(TensorView<TYPE>::init(v),0);
    //}

    //static BGtensor tensor(const initializer_list<initializer_list<TYPE> >& v){
    //return BGtensor(TensorView<TYPE>::init(v),0);
    //}


    static BGtensor batched_tensor(const TENSOR& x){
      return BGtensor(x,1);
    }

    //static BGtensor batched_tensor(const initializer_list<TYPE>& v){
    //return BGtensor(TensorView<TYPE>::init(v),1);
    //}

    //static BGtensor batched_tensor(const initializer_list<initializer_list<TYPE> >& v){
    //return BGtensor(TensorView<TYPE>::init(v),1);
    //}


    static BGtensor grid(const TENSOR& x){
      return BGtensor(x,0,x.ndims());
    }

    static BGtensor grid(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),0,1);
    }

    static BGtensor grid(const initializer_list<initializer_list<TYPE> >& v){
      return BGtensor(TensorView<TYPE>::init(v),0,2);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    //Gdims get_dims() const{
    //return dims.chunk(1+ng);
    //}


  public: // ---- Batches ------------------------------------------------------------------------------------


    bool is_batched() const{
      return dims[0]>1;
    }

    int getb() const{
      return dims[0];
    }

    int bstride() const{
      return strides[0];
    }

    BGtensor batch(const int b) const{
      return BGtensor(arr+b*strides[0],1,gdims(),cdims(),0,gstrides(),cstrides());
    }

    void add_to_batch(const int b, const BGtensor& x) const{
      slice(0,b)+=x.slice(0,0);
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_gridded() const{
      return ng>0;
    }

    bool is_grid() const{
      return ng>0;
    }

    bool has_grid() const{
      return ng>0;
    }

    int ngdims() const{
      return ng;
    }

    Gdims gdims() const{
      return dims.chunk(1,ng);
    }

    Gdims get_gdims() const{
      return dims.chunk(1,ng);
    }

    GstridesB gstrides() const{
      return strides.chunk(1,ng);
    }

    GstridesB get_gstrides() const{
      return strides.chunk(1,ng);
    }


  public: // ---- Cells ---------------------------------------------------------------------------------------------------


    bool has_cells() const{
      return dims.size()>ng+1;
    }

    int ncdims() const{
      return ndims()-ng-1;
    }

    Gdims cdims() const{
      if(!has_cells()) return Gdims();
      return dims.chunk(1+ng);
    }

    Gdims get_cdims() const{
      if(!has_cells()) return Gdims();
      return dims.chunk(1+ng);
    }

    GstridesB cstrides() const{
      if(!has_cells()) return GstridesB();
      return strides.chunk(1+ng);
    }

    int cdim(const int i) const{
      return dims(1+ng+i);
    }

    BGtensor cell(const Gindex& ix) const{
      return BGtensor(arr+get_gstrides().offs(ix),getb(),{},cdims(),bstride(),{},cstrides());
    }

    BGtensor cell(const int b, const Gindex& ix) const{
      return BGtensor(arr+b*strides[0]+get_gstrides().offs(ix),1,{},cdims(),0,{},cstrides());
    }

    BGtensor transp() const{
      CNINE_ASSRT(ncdims()==2);
      auto R(*this);
      R.dims=dims.transp();
      R.strides=strides.transp();
      return R;
    }
 

  public: // ---- Products -------------------------------------------------------------------------------------------------


    #include "BGtensor_products.hpp"


  public: // ---- Operations -----------------------------------------------------------------------------------------------


    BGtensor norm() const{
      BGtensor R(getb(),get_gdims(),Gdims({1,1}),0,get_dev());
      R.add_norm(*this);
      return R;
    }

    void add_norm(const BGtensor& x) const{
      for_each_cell_multi(x,*this,[]
	(const int b, const Gindex& ix, const TENSOR& x, const TENSOR& c){
	  c.set(0,0,x.norm());},1);
    }

    void add_norm_back(const BGtensor& rg, const BGtensor& x) const{
      try{
	for_each_cell_multi(*this,x,rg,[]
	  (const int b, const Gindex& ix, const TENSOR& xg, const TENSOR& x, const TENSOR& rg){
	    xg+=x*(rg(0,0)/x.norm());});
      }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensor::add_norm_back(rg,x): ")+e.what());}
    }


  public: // ---- I/O ------------------------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      oss<<"<BGtensor<"<<TENSOR::dtype_str()<<"> b="<<getb()<<",gdims=";
      oss<<gdims()<<",cdims="<<cdims()<<"> [strides="<<strides<<"]";
      return oss.str();
    }

    string cell_to_string(int b, const Gindex& ix, const string indent="") const{
      if(!has_cells()){
	ostringstream oss;
	oss<<"("<<arr[b*strides[0]+gstrides().offs(ix)]<<")"<<endl;
	return oss.str();
      }
      return cell(b,ix).squeeze(0).str(indent);
    }

    string batch_to_string(int b, const string indent="") const{
      ostringstream oss;
      if(ngdims()==0) 
	oss<<cell_to_string(b,Gindex(),indent);
      else{
	Gdims gdims=get_gdims();
	int ncells=gdims.asize();
	for(int i=0; i<ncells; i++){
	  Gindex ix(i,gdims);
	  oss<<indent<<"Cell "<<ix<<": ";
	  if(has_cells()) oss<<endl;
	  oss<<cell_to_string(b,ix,indent+"  ");
	}
      }
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      if(getb()==1) 
	oss<<batch_to_string(0,indent);
      else{
	for(int i=0; i<getb(); i++){
	  oss<<indent<<"Batch "<<i<<": ";
	  if(ndims()>1) oss<<endl;
	  oss<<batch_to_string(i,indent+"  ");
	}
      }
      return oss.str();
    }

    string str(const string indent="") const{
      return to_string(indent);
    }

    friend ostream& operator<<(ostream& stream, const BGtensor& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE, typename TYPE2, std::enable_if_t<is_numeric_or_complex_v<TYPE2>>* = nullptr>
  inline BGtensor<TYPE> operator*(const TYPE2 c, const BGtensor<TYPE>& x){
    return x*c;
  }

}


#endif 

