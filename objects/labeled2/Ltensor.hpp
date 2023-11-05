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


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "DimLabels.hpp"
#include "TensorSpec.hpp"


namespace cnine{


  template<typename TYPE>
  class Ltensor;

  inline Itensor3_view batch_grid_fused_view3_of(const Ltensor<int>& x);
  inline Rtensor3_view batch_grid_fused_view3_of(const Ltensor<float>& x);
  inline Ctensor3_view batch_grid_fused_view3_of(const Ltensor<complex<float> >& x);



  template<typename TYPE>
  class Ltensor: public TensorView<TYPE>{
  public:

    enum Ttype{batch_grid_cell};

    typedef TensorView<TYPE> BASE;

    //using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    DimLabels labels;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Ltensor(): 
      Ltensor({1},DimLabels(),0,0){}

    Ltensor(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides, const DimLabels& _labels):
      BASE(_arr,_dims,_strides),
      labels(_labels){}

    Ltensor(const Gdims& _dims, const DimLabels& _labels, const int fcode, const int _dev=0):
      BASE(_dims,fcode,_dev), 
      labels(_labels){}


  public: // ---- TensorSpec --------------------------------------------------------------------------------


    Ltensor(const TensorSpec<TYPE>& g):
      Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    static TensorSpec<TYPE> make() {return TensorSpec<TYPE>();}
    static TensorSpec<TYPE> raw() {return TensorSpec<TYPE>().raw();}
    static TensorSpec<TYPE> zero() {return TensorSpec<TYPE>().zero();}
    static TensorSpec<TYPE> sequential() {return TensorSpec<TYPE>().sequential();}
    static TensorSpec<TYPE> gaussian() {return TensorSpec<TYPE>().gaussian();}
    
    TensorSpec<TYPE> spec() const{
      return TensorSpec<TYPE>(dims,labels,dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    Ltensor(const Ltensor& x):
      BASE(x),
      labels(x.labels){}

    Ltensor& operator=(const Ltensor& x){
      BASE::operator=(x);
      labels=x.labels;
    }
    
    Ltensor copy() const{
      Ltensor R(dims,labels,0,dev);
      R=*this;
      return R;
    }

    Ltensor zeros_like() const{
      return Ltensor(dims,labels,0,dev);
    }


  public: // ---- Views -------------------------------------------------------------------------------------


    //auto batch_grid_fused_view1() const -> decltype(batch_grid_fused_view1_of(*this)){
    //CNINE_ASSRT(ndims()==1);
    //return batch_grid_fused_view1_of(*this);
    //}

    //auto batch_grid_fused_view2() const -> decltype(batch_grid_fused_view2_of(*this)){
    //CNINE_ASSRT(ndims()==2);
    //return batch_grid_fused_view2_of(*this);
    //}

    auto bgfused_view3() const -> decltype(batch_grid_fused_view3_of(*this)){
      return batch_grid_fused_view3_of(*this);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Ttype ttype() const{
      return batch_grid_cell;
    }

    bool batch_grid_regular() const{
      return true;
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return labels._batched;
    }

    int nbatch() const{
      CNINE_ASSRT(is_batched());
      return dims[0];
    }

    Ltensor batch(const int i) const{
      CNINE_ASSRT(is_batched());
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return Ltensor(arr+strides[0]*i,dims.chunk(1),strides.chunk(1),labels.copy().set_batched(false));
    }

    void for_each_batch(const std::function<void(const int, const Ltensor& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return labels._narray>0;
    }

    int ngdims() const{
      return labels._narray;
    }

    Gdims gdims() const{
      return labels.gdims(dims);
    }

    GstridesB gstrides() const{
      return labels.gstrides(strides);
    }

    int min_gstride() const{
      if(nbgdims()==0) return 0;
      return strides[nbgdims()-1];
    }


  public: // ---- Batch & Grid  ------------------------------------------------------------------------------


    int nbgdims() const{
      return labels._batched+labels._narray;
    }

    int total_bgdims() const{
      if(nbgdims()==0) return 1;
      return dims[0]*strides[0]/strides[nbgdims()-1];
    }


  public: // ---- Cells --------------------------------------------------------------------------------------


    int ncdims() const{
      return dims.size()-labels._narray-labels._batched;
    }

    Gdims cdims() const{
      return labels.cdims(dims);
    }

    int cdim(const int i) const{
      CNINE_ASSRT(i+nbgdims()<dims.size());
      return dims[nbgdims()+i];
    }

    GstridesB cstrides() const{
      return labels.cstrides(strides);
    }

    int cstride(const int i) const{
      CNINE_ASSRT(i+nbgdims()<dims.size());
      return strides[nbgdims()+i];
    }

    Ltensor cell(const Gindex& ix) const{
      CNINE_ASSRT(!is_batched());
      CNINE_ASSRT(ix.size()==labels._narray);
      return Ltensor(arr+gstrides().offs(ix),cdims(),cstrides(),labels.copy().set_ngrid(0));
    }

    Ltensor cell(const int b, const Gindex& ix) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b<nbatch());
      CNINE_ASSRT(ix.size()==labels._narray+1);
      return Ltensor(arr+strides[0]*b+gstrides().offs(ix),cdims(),cstrides(),labels.copy().set_batched(false).set_ngrid(0));
    }

    void for_each_cell(const std::function<void(const Gindex&, const Ltensor& x)>& lambda) const{
      CNINE_ASSRT(!is_batched());
      gdims().for_each_index([&](const vector<int>& ix){
	  lambda(ix,cell(ix));
	});
    }
    
    void for_each_cell(const std::function<void(const int b, const Gindex&, const Ltensor& x)>& lambda) const{
      CNINE_ASSRT(is_batched());
      for(int b=0; b<nbatch(); b++)
	gdims().for_each_index([&](const vector<int>& ix){
	    lambda(b,ix,cell(b,ix));
	});
    }
    

  public: // ---- Batched cells ------------------------------------------------------------------------------


    Gdims bcdims() const{
      return labels.bcdims(dims);
    }

    GstridesB bcstrides() const{
      return labels.bcstrides(strides);
    }

    Ltensor batched_cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==labels._narray);
      return Ltensor(arr+gstrides().offs(ix),bcdims(),bcstrides(),labels.copy().set_ngrid(0));
    }

    void for_each_batched_cell(const std::function<void(const Gindex&, const Ltensor& x)>& lambda) const{
      gdims().for_each_index([&](const vector<int>& ix){
	  lambda(ix,batched_cell(ix));
	});
    }
    

    

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Ltensor";
    }

    string repr() const{
      ostringstream oss;
      oss<<"Ltensor"<<labels.str(dims)<<"["<<dev<<"]";
      return oss.str();
    }

    string to_string(const string indent="") const{

      if(is_batched()){
	ostringstream oss;
	for_each_batch([&](const int b, const Ltensor& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.to_string(indent+"  ");
	  });
	return oss.str();
      }
      
      if(is_grid()){
	ostringstream oss;
	for_each_cell([&](const Gindex& ix, const Ltensor& x){
	    oss<<indent<<"Cell"<<ix<<":"<<endl;
	    oss<<x.to_string(indent+"  ");
	  });
	return oss.str();
      }
      
      return BASE::str(indent);
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ltensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  // ---- View converters ----------------------------------------------------


  inline Itensor3_view batch_grid_fused_view3_of(const Ltensor<int>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<int>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Itensor3_view(x.mem(),x.total_bgdims(),x.cdim(0),x.cdim(1),x.min_gstride(),x.cstride(0),x.cstride(1),x.dev);
  }

  inline Rtensor3_view batch_grid_fused_view3_of(const Ltensor<float>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<float>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Rtensor3_view(x.mem(),x.total_bgdims(),x.cdim(0),x.cdim(1),x.min_gstride(),x.cstride(0),x.cstride(1),x.dev);
  }

  inline Ctensor3_view batch_grid_fused_view3_of(const Ltensor<complex<float> >& x){
    CNINE_ASSRT(x.ttype()==Ltensor<complex<float> >::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Ctensor3_view(x.arr.ptr_as<float>(),x.arr.ptr_as<float>()+1,
      x.total_bgdims(),x.cdim(0),x.cdim(1),2*x.min_gstride(),2*x.cstride(0),2*x.cstride(1),x.dev);
  }

}

#endif 
