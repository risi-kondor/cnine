
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _MultiLoop
#define _MultiLoop
#include "ThreadGroup.hpp"

namespace cnine{

  extern thread_local int nthreads;


  class MultiLoop{
  public:
    
    MultiLoop(const int n, std::function<void(int)> lambda){

      if(nthreads<=1){
	for(int i=0; i<n; i++) lambda(i);
	return;
      }
      
      ThreadGroup threads(nthreads);
      for(int i=0; i<n; i++)
	threads.add(std::max(1,nthreads/n),lambda,i);

    }

  };

}

#endif 
