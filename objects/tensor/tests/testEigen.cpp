//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "RtensorObj.hpp"
#include "CnineSession.hpp"
#include "SymmetricEigendecomp.hpp"

using namespace cnine;

typedef RtensorObj rtensor;


int main(int argc, char** argv){

  cnine_session session;
  int n=6;

  rtensor T=rtensor::sequential({n,n});
  T.add(T.transp());

  SymmetricEigendecomp solver(T.view2());
  cout<<solver.U<<endl;
  cout<<solver.D<<endl;

}
