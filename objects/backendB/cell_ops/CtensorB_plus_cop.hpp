//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CtensorB_plus_cop
#define _CtensorB_plus_cop

#include "GenericCop.hpp"
#include "Cmaps2.hpp"


namespace cnine{

#ifdef _WITH_CUDA

  //template<typename CMAP>
  //void CtensorB_plus_cu(const CMAP& map, CtensorArrayB& r, const CtensorArrayB& x, const CtensorArrayB& y, 
  //const cudaStream_t& stream, const int add_flag);

  //template<typename CMAP>
  //void CtensorB_plus_accumulator_cu(const CMAP& map, CtensorArrayB& r, const CtensorArrayB& x, const CtensorArrayB& y, const cudaStream_t& stream);

#endif 


  class CtensorB_plus_cop{
  public:

    CtensorB_plus_cop(){}

    void apply(CtensorB& r, const CtensorB& x, const CtensorB& y, const int add_flag=0) const{
      if(add_flag==0){
	r.set(x);
	r.add(y);
      }else{
	r.add(x);
	r.add(y);
      }
    }
 
    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Direct_cmap,CMAP>::value, CMAP>::type>
    void apply(const CMAP& map, CtensorArrayB& r, const CtensorArrayB& x, const CtensorArrayB& y, 
      const int add_flag=0) const{
      // CUDA_STREAM(CtensorA_plus_cu(map,r,x,y,stream,add_flag));
    }

    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Masked2_cmap,CMAP>::value, CMAP>::type>
    void accumulate(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y, const int add_flag=0) const{
      // CUDA_STREAM(CtensorA_plus_accumulator_cu(map,r,x,y,stream));
    }

  };

}

#endif


