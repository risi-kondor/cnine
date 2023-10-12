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
  class Ltensor: public TensorView<TYPE>{
  public:

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

  public: // ---- Access ------------------------------------------------------------------------------------




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


    bool is_gridded() const{
      return labels._narray>0;
    }

    int gdims() const{
      return labels.gdims(dims);
    }

    /*
    int gdims() const{
      return labels.gstrides(strides);
    }

    Ltensor cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==labels._array);
      return Ltensor(arr+gstrides().offs(ix),cell_dims(),cell_strides());
    }

    Ltensor cell(const int i) const{
      CNINE_ASSRT(is_batched());
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return Ltensor(arr+strides[0]*i,dims.chunk(1),strides.chunk(1),labels.copy().set_batched(false));
    }

    void for_each_cell(const std::function<void(const Gindex&, const Ltensor& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }
    */


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
      ostringstream oss;
      if(is_batched())
	for_each_batch([&](const int b, const Ltensor& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.to_string(indent+"  ");
	  });
      else 
	oss<<BASE::str(indent);
      return oss.str();
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


}

#endif 
