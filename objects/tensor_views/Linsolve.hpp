//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineLinsolve
#define _CnineLinsolve

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"
#include "RtensorObj.hpp"

namespace cnine{

#ifdef _WITH_EIGEN
  extern RtensorObj eigen_linsolve(const Rtensor2_view& A, const Rtensor2_view& b);
#endif


  class Linsolve{
  public:

    typedef RtensorObj rtensor;

    rtensor operator()(const Rtensor2_view& A, const Rtensor1_view& b){
#ifdef _WITH_EIGEN
      return eigen_linsolve(A,b);
#endif
    }

    
  };

}

#endif 
