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


#ifndef _CnineEinsum2
#define _CnineEinsum2

#include "Ltensor.hpp"
#include "EinsumForm2.hpp"
#include "EinsumParams.hpp"


namespace cnine{




  class Einsum2{
  public:

    EinsumForm2 form;

    Einsum2(const string str):
      form(str){}

    template<typename TYPE>
    TensorView<TYPE> operator()(const TensorView<TYPE>& x, const TensorView<TYPE>& y, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.bcast_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }
      for(int i=0; i<form.y_ids.size(); i++){
	if(dimensions[form.y_ids[i]]==-1)
	  dimensions[form.y_ids[i]]=y.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.y_ids[i]]==y.dims[i]);
      }

      for(int i=0; i<form.bcast_ids.size(); i++)
	dimensions[form.bcast_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});
      TensorView<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x,y);
      return R;
    }


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){

      auto& transfer_indices=form.transfer_indices;
      auto& xsummation_indices=form.x_summation_indices;
      auto& ysummation_indices=form.y_summation_indices;
      auto& contraction_indices=form.contraction_indices;
      auto& triple_contraction_indices=form.triple_contraction_indices;
      auto& broadcast_indices=form.broadcast_indices;

      CNINE_ASSRT(transfer_indices.size()<=3);
      CNINE_ASSRT(xsummation_indices.size()<=3);
      CNINE_ASSRT(ysummation_indices.size()<=3);
      CNINE_ASSRT(contraction_indices.size()<=3);
      CNINE_ASSRT(triple_contraction_indices.size()<=1);
      CNINE_ASSRT(broadcast_indices.size()<=3);

      Einsum2params params;
      for(int i=0; i<transfer_indices.size(); i++){
	params.tdims[i]=r.dims[transfer_indices[i][2][0]];
	params.tstride_x[i]=x.strides.combine(transfer_indices[i][0]);
	params.tstride_y[i]=y.strides.combine(transfer_indices[i][1]);
	params.tstride_r[i]=r.strides.combine(transfer_indices[i][2]);
      }
      for(int i=0; i<xsummation_indices.size(); i++){
	params.xsdims[i]=x.dims[xsummation_indices[i][0]];
	params.xsstride[i]=x.strides.combine(xsummation_indices[i]);
      }
      for(int i=0; i<ysummation_indices.size(); i++){
	params.ysdims[i]=y.dims[ysummation_indices[i][0]];
	params.ysstride[i]=y.strides.combine(ysummation_indices[i]);
      }
      for(int i=0; i<contraction_indices.size(); i++){
	params.cdims[i]=x.dims[contraction_indices[i].first[0]];
	params.cstride_x[i]=x.strides.combine(contraction_indices[i].first);
	params.cstride_y[i]=y.strides.combine(contraction_indices[i].second);
      }
      for(int i=0; i<triple_contraction_indices.size(); i++){
	params.xdims[i]=r.dims[triple_contraction_indices[i][2][0]];
	params.xstride_x[i]=x.strides.combine(triple_contraction_indices[i][0]);
	params.xstride_y[i]=y.strides.combine(triple_contraction_indices[i][1]);
	params.xstride_r[i]=r.strides.combine(triple_contraction_indices[i][2]);
      }
      for(int i=0; i<broadcast_indices.size(); i++){
	params.bdims[i]=r.dims[broadcast_indices[i][0]];
	params.bstride[i]=r.strides.combine(broadcast_indices[i]);
      }
      
      add_einsum(r,x,y,params);
    }


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& _r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const Einsum2params& p){

      auto& r=const_cast<TensorView<TYPE>& >(_r);

      int xoffs=0;
      int yoffs=0;
      int roffs=0;

      for(int t0=0; t0<p.tdims[0]; t0++)
	for(int t1=0; t1<p.tdims[1]; t1++)
	  for(int t2=0; t2<p.tdims[2]; t2++){
	    int xoffs_t=xoffs+t0*p.tstride_x[0]+t1*p.tstride_x[1]+t2*p.tstride_x[2];
	    int yoffs_t=yoffs+t0*p.tstride_y[0]+t1*p.tstride_y[1]+t2*p.tstride_y[2];
	    int roffs_t=roffs+t0*p.tstride_r[0]+t1*p.tstride_r[1]+t2*p.tstride_r[2];

	    TYPE t=0;
	    for(int c0=0; c0<p.cdims[0]; c0++)
	      for(int c1=0; c1<p.cdims[1]; c1++)
		for(int c2=0; c2<p.cdims[2]; c2++){
		  int xoffs_c=xoffs_t+c0*p.cstride_x[0]+c1*p.cstride_x[1]+c2*p.cstride_x[2];
		  int yoffs_c=yoffs_t+c0*p.cstride_y[0]+c1*p.cstride_y[1]+c2*p.cstride_y[2];

		  TYPE xt=0;
		  for(int xs0=0; xs0<p.xsdims[0]; xs0++)
		    for(int xs1=0; xs1<p.xsdims[1]; xs1++)
		      for(int xs2=0; xs2<p.xsdims[2]; xs2++)
			xt+=*(x.get_arr()+xoffs_c+xs0*p.xsstride[0]+xs1*p.xsstride[1]+xs2*p.xsstride[2]);

		  TYPE yt=0;
		  for(int ys0=0; ys0<p.ysdims[0]; ys0++)
		    for(int ys1=0; ys1<p.ysdims[1]; ys1++)
		      for(int ys2=0; ys2<p.ysdims[2]; ys2++)
			yt+=*(y.get_arr()+yoffs_c+ys0*p.ysstride[0]+ys1*p.ysstride[1]+ys2*p.ysstride[2]);

		  t+=xt*yt;
		}

	    for(int b0=0; b0<p.bdims[0]; b0++)
	      for(int b1=0; b1<p.bdims[1]; b1++)
		for(int b2=0; b2<p.bdims[2]; b2++)
		  *(r.get_arr()+roffs_t+b0*p.bstride[0]+b1*p.bstride[1]+b2*p.bstride[2])+=t;
	  }

    }
    

  };

}

#endif 
