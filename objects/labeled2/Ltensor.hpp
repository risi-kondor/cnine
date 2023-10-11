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


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Cnine_base.hpp"
#include "LtensorView.hpp"
#include "LtensorGen.hpp"


namespace cnine{

  template<typename TYPE>
  class Ltensor: public LtensorView<TYPE>{
  public:

    typedef LtensorView<TYPE> BASE;

    using BASE::BASE;

    using LtensorView<TYPE>::dims;
    using LtensorView<TYPE>::strides;
    using LtensorView<TYPE>::dev;

    //using LtensorView<TYPE>::operator=;
    using LtensorView<TYPE>::ndims;
    using LtensorView<TYPE>::dim;
    using LtensorView<TYPE>::set;
    using LtensorView<TYPE>::transp;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Ltensor(): 
      Ltensor({1},DimLabels(),0,0){}

    Ltensor(const NewTensor& g):
      Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    /*
    Ltensor(const Ltensor<TYPE>& x):
      BASE(x.dims,x.strides), labels(x.labels){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    Ltensor(const Ltensor<TYPE>& x, const nowarn_flag& dummy):
      BASE(x.dims,x.dev){
      view()=x.view();
    }
        
    Ltensor(Ltensor<TYPE>&& x):
      TensorView<TYPE>(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
      }
    */

  };
}

#endif 


    //template<typename FILLTYPE, typename = typename 
    //std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ltensor(const Gdims& _dims, const DimLabels& _labels, const FILLTYPE& fill, const int _dev=0):
    //BASE(_dims,_labels,fill,_dev){}

    //Ltensor(const Gdims& _dims, const DimLabels& _labels, const int fcode, const int _dev){
    //switch(fcode){
    //default: 
    //}
