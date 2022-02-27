//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Ctensor1view_add
#define _Ctensor1view_add

//#include "GenericCop.hpp"


namespace cnine{

  //#ifdef _WITH_CUDA
//template<typename CMAP>
  //void CtensorA_add_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);
  //#endif 


  class Ctensor1view_add{
  public:

    Ctensor1view_add(){}
    
    void apply(Ctensor1_view& r, const Ctensor1_view& x, const float v, const bool add_flag=true) const{
      r.add(x,v);
    }

    /*
    template<typename IMAP>
    void operator()(const IMAP& map, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      CtensorA_add_cu(map,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }
    */

  };


}

#endif
