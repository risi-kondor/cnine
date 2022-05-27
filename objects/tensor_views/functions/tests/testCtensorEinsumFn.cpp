//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorB.hpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "CtensorEinsumFn.hpp"

using namespace cnine;

typedef RtensorA rtensor; 
typedef CtensorB ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensor X=ctensor::gaussian({4,4});
  ctensor Y=ctensor::gaussian({4,4});
  ctensor R=ctensor::zeros({4,4});

  CtensorEinsumFn<float> fn("ij,ji->ab");
  fn(R.viewx(),X.viewx(),Y.viewx());

  PRINTL(R);

}
