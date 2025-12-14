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

#ifndef _ForEachCellMulti
#define _ForEachCellMulti

#include "BGtensor.hpp"    


namespace cnine{

  template<typename TYPE>
  class BGtensor;


  template<typename TYPE>
  class ForEachCellMulti{
  public: 


    void operator()(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y, const BGtensor<TYPE>& z, 
		    std::function<void(const int b, const Gindex& cell,
				       const TensorView<TYPE>& x,  const TensorView<TYPE>& y, const TensorView<TYPE>& z)> lambda,
		    const int target=0) const{
    
      int B=x.dominant_batch(x,y,z);
      Gdims gdims=x.dominant_gdims(x,y,z);
      int ncells=gdims.asize();
      int ngdims=gdims.size();
      bool sequential=(target==0 && x.getb()==1)||(target==1 && y.getb()==1)||(target==2 && z.getb()==1);

      // Shortcut
      if(ngdims==0){ 
	Gindex null_ix;
	MultiLoop(B,[&](const int b){
		    lambda(b,null_ix,x.slice(0,(x.dim(0)>1)*b),y.slice(0,(y.dim(0)>1)*b),z.slice(0,(z.dim(0)>1)*b));}
		  ,sequential);
	return;
      }

      TensorView<TYPE> xcell(x.arr,x.get_cdims(),x.cstrides());
      GstridesB x_gstrides=GstridesB::zero(ngdims);
      if(x.has_grid()) x_gstrides=x.gstrides();
      int x_bstride=x.strides[0]*(x.is_batched());

      TensorView<TYPE> ycell(y.arr,y.get_cdims(),y.cstrides());
      GstridesB y_gstrides=GstridesB::zero(ngdims);
      if(y.has_grid()) y_gstrides=y.gstrides();
      int y_bstride=y.strides[0]*(y.is_batched());

      TensorView<TYPE> zcell(z.arr,z.get_cdims(),z.cstrides());
      GstridesB z_gstrides=GstridesB::zero(ngdims);
      if(z.has_grid()) z_gstrides=z.gstrides();
      int z_bstride=z.strides[0]*(z.is_batched());

      MultiLoop(B,[&](const int b){
		  for(int i=0; i<ncells; i++){
		    Gindex ix(i,gdims);
		    xcell.arr=x.arr+x_bstride*b+x_gstrides.offs(ix);
		    ycell.arr=y.arr+y_bstride*b+y_gstrides.offs(ix);
		    zcell.arr=z.arr+z_bstride*b+z_gstrides.offs(ix);
		    lambda(b,ix,xcell,ycell,zcell);
		  }
		},sequential);

    }
    
  };

}

#endif 
