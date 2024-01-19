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

#ifndef _cnine_gather_rows
#define _cnine_gather_rows

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "WeightedGatherMapB.hpp"
#include "FixedkGatherMap.hpp"
#include "Ltensor.hpp"
#include "logged_timer.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  extern void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const GatherMapB& g, const cudaStream_t& stream);
  extern void gatherRowsw_cu(const Rtensor2_view& r, const Rtensor2_view& x, const WeightedGatherMapB& g, const cudaStream_t& stream);
  extern void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const FixedkGatherMap& g, const cudaStream_t& stream);
#endif 
  
  class GatherRows{
  public:

    template<typename TYPE>
    void operator()(TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const GatherMapB& g){
      CNINE_ASSRT(_r.ndims()==2);
      CNINE_ASSRT(_r.dim(1)%g.out_columns==0);
      CNINE_ASSRT(_x.ndims()==2);
      CNINE_ASSRT(_x.dim(1)%g.in_columns==0);

      if(g.fixedk_maps.size()>0){
	for(auto& p: g.fixedk_maps)
	  (*this)(_r,_x,*p);
      }

      if(dynamic_cast<const WeightedGatherMapB*>(&g)) 
	weighted(_r,_x,dynamic_cast<const WeightedGatherMapB&>(g));

      if(g.size()==0) return;

      auto r=_r.view2();
      r.n0*=g.out_columns;
      r.n1=r.n1*g.out_columns_n/g.out_columns;
      r.s0/=g.out_columns;
      auto x=_x.view2();
      x.n0*=g.in_columns;
      x.n1=x.n1*g.in_columns_n/g.in_columns;
      x.s0/=g.in_columns;
      CNINE_ASSRT(r.n1==x.n1);
    
      if(_r.get_dev()==0){
	fnlog timer("GatherRows::operator()");
	logged_timer ptimer("GatherRows(CPU)",r,x,((long long)g.n_ops())*x.n1);
	CNINE_ASSRT(g.get_dev()==0);
	int N=g.size();
	for(int i=0; i<N; i++){
	  auto targt=r.slice0(g.target(i));
	  //targt.n0=x.n1; // hack // change in cu too 
	  int M=g.size_of(i);
	  for(int j=0; j<M; j++){
	    targt+=x.slice0(g(i,j));
	  }
	}
      }

      if(_r.get_dev()==1){
	g.sort();
	fnlog timer("GatherRows::operator()(G)");
	logged_timer ptimer("GatherRows(GPU)",r,x,((long long)g.n_ops())*x.n1);
	CUDA_STREAM(gatherRows_cu(r,x,g,stream));
      }
    }


    template<typename TYPE>
    void weighted(TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const WeightedGatherMapB& g){
      auto r=_r.view2();
      r.n0*=g.out_columns;
      r.n1/=g.out_columns;
      r.s0/=g.out_columns;
      auto x=_x.view2();
      x.n0*=g.in_columns;
      x.n1/=g.in_columns;
      x.s0/=g.in_columns;

      if(_r.get_dev()==0){
	fnlog timer("GatherRows::weighted()");
	logged_timer ptimer("GatherRows::weighted(CPU)",r,x,((long long)g.n_ops())*x.n1);
	CNINE_ASSRT(g.get_dev()==0);
	int N=g.size();
	for(int i=0; i<N; i++){
	  auto targt=r.slice0(g.target(i));
	  targt.n0=x.n1; // hack
	  int M=g.size_of(i);
	  for(int j=0; j<M; j++)
	    targt.add(x.slice0(g.src(i,j)),g.weight(i,j));
	}
      }

      if(_r.get_dev()==1){
	g.sort();
	fnlog timer("GatherRows::weighted()(G)");
	logged_timer ptimer("GatherRows::weighted(GPU)",r,x,((long long)g.n_ops())*x.n1);
	CUDA_STREAM(gatherRowsw_cu(r,x,g,stream));
      }
    }



  template<typename TYPE>
  void operator()(TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const FixedkGatherMap& g){
    CNINE_ASSRT(_r.ndims()==2);
    CNINE_ASSRT(_r.dim(0)%g.out_columns==0);
    CNINE_ASSRT(_x.ndims()==2);
    CNINE_ASSRT(_x.dim(0)%g.in_columns==0);

    auto r=_r.view2();
    r.n0/=g.out_columns;
    r.n1*=g.out_columns;
    auto x=_x.view2();
    x.n0/=g.in_columns;
    x.n1*=g.in_columns;
    
    if(_r.get_dev()==0){
      CNINE_ASSRT(g.get_dev()==0);
      int N=g.getn();
      int K=g.getk();
      for(int i=0; i<N; i++){
	int targt=g.target(i);
	for(int j=0; j<K; j++)
	  r.slice0(targt)+=x.slice0(g(i,j));
      }
    }

    if(_r.get_dev()==1){
      CUDA_STREAM(gatherRows_cu(r,x,g,stream));
    }
  }


  template<typename TYPE>
  Ltensor<TYPE> operator()(const TensorView<TYPE>& x, const GatherMapB& g){
    CNINE_ASSRT(x.ndims()==2);
    Ltensor<TYPE> r({g.getn(),x.dim(1)},0,x.get_dev());
    (*this)(r,x,g);
    return r;
  }

  template<typename TYPE>
  Ltensor<TYPE> operator()(const TensorView<TYPE>& x, const FixedkGatherMap& g){
    CNINE_ASSRT(x.ndims()==2);
    Ltensor<TYPE> r({g.getn(),x.dim(1)},0,x.get_dev());
    (*this)(r,x,g);
    return r;
  }

  };


  class MultiGatherRows{
  public:
    
    template<typename PACK>
    void operator()(PACK& _r, const PACK& _x, const GatherMapB& g){
    }    

  };

}

#endif 
