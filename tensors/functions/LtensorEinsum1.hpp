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


#ifndef _CnineLtensorEinsum1
#define _CnineLtensorEinsum1

#include "Ltensor.hpp"


namespace cnine{


  struct EsumParams{
  public:

    int ddims[4];
    int xstride_d[4];
    int rstride_d[4];

    int sdims[4];
    int xstride_s[4];

    int bdims[4];
    int rstride_b[4];
  };

  extern void LtensorEinsum1loops(int d, int r , int b, float* R, const float* x, const EsumParams& params);


  class LtensorEinsum1{
  public:

    vector<pair<vector<int>,vector<int> > > dstrides;
    vector<vector<int> > sstrides;
    vector<vector<int> > bstrides;
    vector<int> r_ids;
    vector<int> x_ids;
    vector<int> bcast_ids;
    int id_tail=0;

    LtensorEinsum1(const string str){

      auto d1=str.find("->");
      if(d1==string::npos){
	COUT("Error in RtensorEinsumFn: malformed einsum string");
	return;
      }
      auto xstr=str.substr(0,d1);
      auto rstr=str.substr(d1+2,string::npos);
      x_ids=vector<int>(xstr.size());
      r_ids=vector<int>(rstr.size());
      cout<<xstr<<endl;
      cout<<rstr<<endl;

      while(true){
	auto p=rstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=rstr[p];
	auto v=find_all(rstr,c);
	for(auto q:v) r_ids[q]=id_tail;
	if(xstr.find(c)==string::npos){
	  bstrides.push_back(v);
	  bcast_ids.push_back(id_tail);
	}else{
	  auto w=find_all(xstr,c);
	  for(auto q:w) x_ids[q]=id_tail;
	  dstrides.push_back(make_pair(v,w));
	}
	id_tail++;
      }

      while(true){
	auto p=xstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=xstr[p];
	auto v=find_all(xstr,c);
	for(auto q:v) x_ids[q]=id_tail;
	sstrides.push_back(v);
	id_tail++;
      }
    }

    template<typename TYPE>
    Ltensor<TYPE> operator()(const Ltensor<TYPE>& x, vector<int> rdims={}){
      //CNINE_ASSRT(rdims.size()==bcast_ids.size());
      
      vector<int> dimensions(id_tail,-1);
      for(int i=0; i<x_ids.size(); i++){
	if(dimensions[x_ids[i]]==-1)
	  dimensions[x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[x_ids[i]]==x.dims[i]);
      }

      for(int i=0; i<bcast_ids.size(); i++)
	if(i<rdims.size())
	  dimensions[bcast_ids[i]]=rdims[i];
	else
	  dimensions[bcast_ids[i]]=3;

      auto r_dims=mapcar<int,int>(r_ids,[&](const int& id){return dimensions[id];});
      Ltensor<TYPE> R(r_dims,0,x.get_dev());

      add_einsum(R,x);

      return R;
    }
      


    template<typename TYPE>
    void add_einsum(const Ltensor<TYPE>& r, const Ltensor<TYPE>& x){

      CNINE_ASSRT(dstrides.size()<=4);
      CNINE_ASSRT(sstrides.size()<=4);
      CNINE_ASSRT(bstrides.size()<=4);

      EsumParams params;
      for(int i=0; i<dstrides.size(); i++){
	params.ddims[i]=r.dims[dstrides[i].first[0]];
	params.rstride_d[i]=r.strides.combine(dstrides[i].first);
	params.xstride_d[i]=x.strides.combine(dstrides[i].second);
      }
      for(int i=0; i<sstrides.size(); i++){
	params.sdims[i]=x.dims[sstrides[i][0]];
	params.xstride_s[i]=x.strides.combine(sstrides[i]);
      }
      for(int i=0; i<bstrides.size(); i++){
	params.bdims[i]=r.dims[bstrides[i][0]];
	params.rstride_b[i]=r.strides.combine(bstrides[i]);
      }

      LtensorEinsum1loops(dstrides.size(),sstrides.size(),bstrides.size(),const_cast<TYPE*>(r.get_arr()), x.get_arr(), params);
    }

   
    /*
    void compute(TYPE* r, TYPE* x, const EsumParams& params, const int n_direct, const int n_sum, const int n_bcast){

      switch(n_direct){
      case 0:
	compute_sub(r,x,params,n_sum,n_bcast);
	break;
      case 1:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  compute_sub(r+dstridesr[0]*i0,x+dstridesx[0]*i0,params,n_sum,n_bcast);
	break;
      case 2:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1,x+dstridesx[0]*i0+dstridesx[1]*i1,params,n_sum,n_bcast);
	break;
      case 3:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    for(int i2=0; i2<params.ddims[2]; i2++)
	      compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1+dstridesr[2]*i2,
		x+dstridesx[0]*i0+dstridesx[1]*i1+dstridesx[2]*i2,params,n_sum,n_bcast);
	break;
      case 4:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    for(int i2=0; i2<params.ddims[2]; i2++)
	      compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1+dstridesr[2]*i2+dstridesr[3]*i3,
		x+dstridesx[0]*i0+dstridesx[1]*i1+dstridesx[2]*i2+dstridesx[3]*i3,params,n_sum,n_bcast);
	break;
      }
      }
    */
    
  private:

    inline vector<int> find_all(string& str, const char c) const{
      vector<int> r;
      auto p=str.find_first_of(c);
      while(p!=string::npos){
	str.replace(p,1,1,'x');
	r.push_back(p);
	p=str.find_first_of(c);
      }
      return r;
    }

  };

}

#endif 
