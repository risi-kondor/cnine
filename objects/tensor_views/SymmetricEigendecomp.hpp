//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineSymmetricEigendecomp
#define _CnineSymmetricEigendecomp

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"
#include "RtensorObj.hpp"

namespace cnine{

#ifdef _WITH_EIGEN
  extern pair<RtensorObj,RtensorObj> eigen_eigendecomp(const Rtensor2_view& x);
#endif


  class SymmetricEigendecomp{
  public:

    typedef RtensorObj rtensor;

    rtensor U;
    rtensor D;

    SymmetricEigendecomp(const Rtensor2_view& x){
#ifdef _WITH_EIGEN
      auto p=eigen_eigendecomp(x);
      U=p.first;
      D=p.second;
#endif
    }

    
  };

}

#endif 
