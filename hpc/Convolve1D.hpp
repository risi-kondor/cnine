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

#ifndef _CnineConvolve1D
#define _CnineConvolve1D

#include "TensorView.hpp"
#include "MultiLoop.hpp"


namespace cnine{

  class Convolve1D{
  public:


    template<typename TYPE, typename WTYPE>
    void operator()(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<WTYPE>& w){
      int dev=r.get_dev();
      int d=r.ndims();
      CNINE_ASSRT(x.ndims()==d);
      int wx=w.dim(0);

      if(d==1){ // (i0)*(j0) -> (i0+j0)
	CNINE_ASSRT(w.ndims()==1);
	if(dev==0){
	  int pad0=(r.dim(0)-x.dim(0)+w.dim(0)-1)/2;
	  for(int i0=0; i0<r.dim(0); i0++){
	    TYPE t=0;
	    for(int j0=std::max(0,pad0-i0); j0<std::min(wx,x.dim(0)-i0+pad0); j0++)
	      t+=x(i0+j0-pad0)*w(j0);
	    r.set(i0,t);
	  }
	}
      }

      if(d==2){ // (i0,a)*(j0,a'a) -> (i0+j0,a')
	CNINE_ASSRT(w.ndims()==3);
	if(dev==0){
	  int pad0=(r.dim(0)-x.dim(0)+w.dim(0)-1)/2;
	  for(int i0=0; i0<r.dim(0); i0++){
	    auto R=r.slice(0,i0);
	    for(int j0=std::max(0,pad0-i0); j0<std::min(wx,x.dim(0)-i0+pad0); j0++){
	      auto W=w.slice(0,j0);
	      auto X=x.slice(0,i0+j0-pad0);
	      R.add_mvprod(W,X);
	    }
	  }
	}
      }

      if(d==3){ // (i0,a,c)*(j0,a',a) -> (i0+j0,a',c) 
	CNINE_ASSRT(w.ndims()==3);
	if(dev==0){
	  int pad0=(r.dim(0)-x.dim(0)+w.dim(0)-1)/2;
	  for(int i0=0; i0<r.dim(0); i0++){
	    auto R=r.slice(0,i0);
	    for(int j0=std::max(0,pad0-i0); j0<std::min(wx,x.dim(0)-i0+pad0); j0++){
	      auto W=w.slice(0,j0);
	      auto X=x.slice(0,i0+j0-pad0);
	      R.add_mprod(W,X);
	    }
	  }
	}
      }

      if(d==5){ // (b,i0,a,c)*(j0,a',a) -> (b,i0+j0,a',c)
	if(dev==0){
	  CNINE_ASSRT(r.dim(0)==x.dim(0));
	  cnine::MultiLoop(r.dim(0),[&](const int b){
			     (*this)(r.slice(0,b),x.slice(0,b),w);});
	}
      }
    }

  };


  
  class Convolve1D_back0{
  public:

    template<typename TYPE, typename WTYPE>
    void operator()(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<WTYPE>& w){

      if(w.ndims()==1){
	TensorView<WTYPE> _w(w.arr+(w.dim(0)-1)*w.stride(0),{w.dim(0)},{-w.stride(0)});
	Convolve1D()(x,r,_w);
	return;
      }

      CNINE_ASSRT(w.ndims()==3);
      TensorView<WTYPE> _w(w.arr+(w.dim(0)-1)*w.stride(0),
			   {w.dim(0),w.dim(2),w.dim(1)},
			   {-w.stride(0),w.stride(2),w.stride(1)});
      Convolve1D()(x,r,_w);
    }
  };
  

  class Convolve1D_back1{
  public:

    template<typename TYPE, typename WTYPE>
    void operator()(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<WTYPE>& w){
      int d=r.ndims();
      CNINE_ASSRT(x.ndims()==d);
      if(d==1){
	Convolve1D()(w,x,r);
	return;
      }
      CNINE_ASSRT(w.ndims()==3);
      if(d==2) Convolve1D()(w,x.unsqueeze(1),r.unsqueeze(2));
      if(d==3) Convolve1D()(w,x.transp(1,2),r);
      if(d==4){
	int b=x.dim(0);
	MultiLoop(b,[&](const int b){
		    (*this)(r.slice(0,b),x.slice(0,b),w);});
      }
    }
  };

    
  template<typename TYPE, typename TYPE2>
  inline TensorView<TYPE> convolve1d(const TensorView<TYPE>& x, const TensorView<TYPE2>& w, int pad=0){
    int dev=x.get_dev();
    int d=x.ndims();
    int I0=x.dim(0)+2*pad-w.dim(0)+1;
    if(pad==-1) I0=x.dim(0); // note: for even size filters this offsets the grid by 0.5 to the right

    if(d==1){
      TensorView<TYPE> r({I0},0,dev);
      Convolve1D()(r,x,w);
      return r;
    }

    CNINE_ASSRT(w.ndims()==3);
    int aout=w.dim(-2);
    if(d==2){
      TensorView<TYPE> r({I0,aout},0,dev);
      Convolve1D()(r,x,w);
      return r;
    }
    if(d==3){
      TensorView<TYPE> r({I0,aout,x.dim(2)},0,dev);
      Convolve1D()(r,x,w);
      return r;
    }
    if(d==4){
      TensorView<TYPE> r({I0,aout,x.dim(3)},0,dev);
      Convolve1D()(r,x,w);
      return r;
    }
    return TensorView<TYPE>();
  }

}


#endif 
