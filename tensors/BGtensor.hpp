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
      return BGtensor(slice(0,b),0);
    }

    void add_to_batch(const int b, const BGtensor& x) const{
      slice(0,b)+=x.slice(0,0);
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_gridded() const{
      return ng>1;
    }

    bool is_grid() const{
      return ng>1;
    }

    bool has_grid() const{
      return ng>1;
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

    TENSOR cell(const int b, const Gindex& ix) const{
      return TENSOR(arr+b*strides[0]+get_gstrides().offs(ix),cdims(),cstrides());
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
      return cell(b,ix).str(indent);
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


  template<typename TYPE, typename TYPE2>
  inline BGtensor<TYPE> operator*(const TYPE2 c, const BGtensor<TYPE>& x){
    return x*c;
  }

}


#endif 

    /*
    template<typename TYPE2>
    static Gdims dominant_gdims(const BGtensor& x, const BGtensor<TYPE2>& y){
      Gdims xg=x.gdims();
      Gdims yg=y.gdims();
      if(xg==yg) return xg;
      if(!x.is_grid()) return yg;
      if(!y.is_grid()) return xg;
      throw std::invalid_argument("Genet error: the grid dimensions of "+x.repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return Gdims();
    }

    static Gdims dominant_gdims(const BGtensor& x, const BGtensor& y, const BGtensor& z){
      Gdims xg=x.gdims();
      Gdims yg=y.gdims();
      Gdims zg=y.gdims();

      int ng=(int)(xg.size()>0)+(int)(yg.size()>0)+(int)(zg.size()>0);
      if(ng==0) return Gdims();
      if(ng==1){
	if(xg.size()>0) return xg;
	if(yg.size()>0) return yg;
	return zg;
      }
      if(ng==2){
	if(xg.size()==0){
	  CNINE_ASSRT(yg==zg);
	  return yg;
	}
	if(yg.size()==0){
	  CNINE_ASSRT(xg==zg);
	  return xg;
	}
	if(zg.size()==0){
	  CNINE_ASSRT(xg==yg);
	  return xg;
	}
      }
      CNINE_ASSRT(xg==yg && xg==zg);
      return xg;
    }
    */
   /*
    template<typename TYPE2>
    static Gdims dominant_cdims(const BGtensor& x, const BGtensor<TYPE2>& y){
      Gdims xg=x.get_cdims();
      Gdims yg=y.get_cdims();
      if(xg==yg) return xg;
      if(!x.has_cells()) return yg;
      if(!y.has_cells()) return xg;
      throw std::invalid_argument("Genet error: the cell dimensions of "+x.repr()+" and "+y.repr()+" cannot be reconciled.");
      return Gdims();
    }
    */
