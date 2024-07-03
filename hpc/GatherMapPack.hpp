/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GatherMapPack
#define _GatherMapPack

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "shared_object_pack.hpp"


namespace cnine{

  class GatherMapPack: public shared_object_pack<GatherMapB>{
  public:

    typedef shared_object_pack<GatherMapB> BASE;

    int n=0;
    int m=0;

    int in_columns=1;
    int out_columns=1;
    int in_columns_n=1;
    int out_columns_n=1;

    Ltensor<int> in_offsets;
    Ltensor<int> out_offsets;

    GatherMapPack(const std::vector<shared_ptr<GatherMapB> >& v):
      BASE(v){
      int N=v.size();
      CNINE_ASSRT(N>0);
      in_offsets.reset(Ltensor<int>(Gdims(N)));
      out_offsets.reset(Ltensor<int>(Gdims(N)));
      
      m=v[0]->n;
      in_columns=v[0]->in_columns;
      out_columns=v[0]->out_columns;
      in_columns_n=v[0]->in_columns_n;
      out_columns_n=v[0]->out_columns_n;

      int in_offs=0;
      int out_offs=0;
      for(int i=0; i<N; i++){
	CNINE_ASSRT(v[i]->m==m);
	CNINE_ASSRT(v[i]->in_columns==in_columns);
	CNINE_ASSRT(v[i]->out_columns==out_columns);
	CNINE_ASSRT(v[i]->in_columns_n==in_columns_n);
	CNINE_ASSRT(v[i]->out_columns_n==out_columns_n);
	in_offsets(i)=in_offs;
	out_offsets(i)=out_offs;
	in_offs+=v[i]->m*in_columns;
	out_offs+=v[i]->n*out_columns;
	n+=v[i]->n;
	m+=v[i]->m;
      }

    }

    const GatherMapPack& sort() const{
      for(auto& p: *this)
	p->sort();
      return *this;
    }

  };

}

#endif 
