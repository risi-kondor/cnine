//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _TensorView_accumulator
#define _TensorView_accumulator

#include "Cnine_base.hpp"
#include "Rmask1.hpp"
#include "Ctensor2_view.hpp"

namespace cnine{

    #ifdef _WITH_CUDA
    void Ctensor2view_accumulator_cu(Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask, const cudaStream_t& stream);
    #endif 

class Ctensor2view_accumulator{
    public:

    Ctensor2view_accumulator(Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask){
      if(r.dev==0){
	    assert(x.dev==0);
	    for(auto it: mask.lists){
	        auto t=r.slice0(it.first);
	        auto& lst=it.second;
	        for(int i=0; i<lst.size(); i++)
	            t.add(x.slice0(lst[i].first),lst[i].second);
	    }
      }
      if(r.dev==1){
#ifdef _WITH_CUDA
    	cudaStream_t stream;
	    CUDA_SAFE(cudaStreamCreate(&stream));
	    Ctensor2view_accumulator_cu(r,x,mask,stream);
	    CUDA_SAFE(cudaStreamSynchronize(stream));
	    CUDA_SAFE(cudaStreamDestroy(stream));
#else
	    CNINE_NOCUDA_ERROR;
#endif
      }
      
    }
};

}

#endif