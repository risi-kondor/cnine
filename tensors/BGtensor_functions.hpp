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

#ifndef _BGtensor_functions
#define _BGtensor_functions

#include "BGtensor.hpp"
#include "ForEachCellMulti.hpp"
#include "ForEachCellMultiScalar.hpp"


namespace cnine{



  // ---- Elementwise products -------------------------------------------------------------------------------------------------

 

}

#endif 


  /*
  template<typename TYPE>
  inline void add_prod(const BGtensor<TYPE>& r, const BGtensor<TYPE>& x, const BGtensor<TYPE>& y){

    if(x.nc==0){
      if(y.nc==0) CNINE_UNIMPL();
      ForEachCellMultiScalar<TYPE>()(r,y,x,[](const int b, const Gindex& ix,
	  const TensorView<TYPE>& r, const TensorView<TYPE>& y, const TYPE c){
	  r.add(y,c);},0);
      return;
    }

    if(y.nc==0){
      ForEachCellMultiScalar<TYPE>()(r,x,y,[](const int b, const Gindex& ix,
	  const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TYPE c){
	  r.add(x,c);},0);
      return;
    }

    ForEachCellMulti<TYPE>()(r,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_prod(x,y);},0);
  }
  */
  /*
  template<typename TYPE>
  inline BGtensor<TYPE> operator*(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y){
    BGtensor<TYPE> R(x.dominant_batch(x,y),x.dominant_gdims(x,y),x.dominant_cdims(x,y),0,x.get_dev());
    R.add_prod(x,y);
    return R;
  }


  template<typename TYPE>
  inline BGtensor<TYPE> oprod(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y){
    BGtensor<TYPE> R(x.dominant_batch(x,y),x.dominant_gdims(x,y),x.dominant_cdims(x,y),0,x.get_dev());
    R.add_prod(x,y);
    return R;
  }
  */
 
