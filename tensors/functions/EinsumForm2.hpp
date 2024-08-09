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


#ifndef _CnineEinsumForm2
#define _CnineEinsumForm2

#include "Ltensor.hpp"


namespace cnine{


  class EinsumForm2: EinsumFormBase{
  public:

    vector<vector<vector<int> > > transfer_indices;
    vector<vector<int> > x_summation_indices;
    vector<vector<int> > y_summation_indices;
    vector<pair<vector<int>,vector<int> > > contraction_indices;
    vector<vector<int> > broadcast_indices;
    vector<int> x_ids;
    vector<int> y_ids;
    vector<int> r_ids;
    vector<int> bcast_ids;
    int id_tail=0;


    EinsumForm2(const string str){

      auto d0=str.find(",");
      auto d1=str.find("->");
      if(d0==string::npos || d1==string::npos || d0>d1){
	CNINE_ERROR(str+" is not a well formed einsum string.");
	return;
      }
      auto xstr=str.substr(0,d0);
      auto ystr=str.substr(d0+1,d1-d0-1);
      auto rstr=str.substr(d1+2,string::npos);
      x_ids=vector<int>(xstr.size());
      y_ids=vector<int>(xstr.size());
      r_ids=vector<int>(rstr.size());

      while(true){
	
	// find a new index i appearing in the result
	auto p=rstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=rstr[p];
	auto v=find_all(rstr,c);
	for(auto q:v) r_ids[q]=id_tail;

	// if i is a broadcast index
	if(xstr.find(c)==string::npos && ystr.find(c)==string::npos){
	  broadcast_indices.push_back(v);
	  bcast_ids.push_back(id_tail);
	}

	// if i is a transfer index
	else{
	  auto xw=find_all(xstr,c);
	  for(auto q:xw) x_ids[q]=id_tail;
	  auto yw=find_all(ystr,c);
	  for(auto q:yw) y_ids[q]=id_tail;
	  vector<vector<int> > u;
	  u.push_back(xw);
	  u.push_back(yw);
	  u.push_back(v);
	  transfer_indices.push_back(u);
	}

	id_tail++;
      }

      while(true){

	// find a new index i appearing in x
	auto p=xstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=xstr[p];
	auto v=find_all(xstr,c);
	for(auto q:v) x_ids[q]=id_tail;

	// if i is a summation index 
	if(ystr.find(c)==string::npos)
	  x_summation_indices.push_back(v);

	else{
	  auto w=find_all(ystr,c);
	  for(auto q:w) y_ids[q]=id_tail;
	  contraction_indices.push_back(make_pair(v,w));
	}

	id_tail++;
      }

      while(true){
	
	// the remainder are y summation indices
	auto p=ystr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=ystr[p];
	auto v=find_all(ystr,c);
	for(auto q:v) y_ids[q]=id_tail;
	y_summation_indices.push_back(v);
	id_tail++;
      }

    }


    // ---- I/O -----------------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;

      oss<<indent<<"x summations: ";
      for(auto& p:x_summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"y summations: ";
      for(auto& p:y_summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Contractions: ";
      for(auto& p:contraction_indices)
	oss<<p.first<<"*"<<p.second<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Transfers: ";
      for(auto& p:transfer_indices)
	oss<<p[0]<<","<<p[1]<<"->"<<p[2]<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Broadcasting: ";
      for(auto& p:broadcast_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EinsumForm2& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
