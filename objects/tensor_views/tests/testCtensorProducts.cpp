//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorB.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef CtensorB ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensor A=ctensor::sequential(Gdims(3,3,3));
  ctensor B=ctensor::sequential(Gdims(3,3,3));
  ctensor C=ctensor::zero(Gdims(3,3,3,3));

  C.view4().add_expand_2(A.view3(),B.view3());
  print(C);


  #ifdef _WITH_CUDA

  ctensor Ac=A.to_device(1);
  ctensor Bc=B.to_device(1);
  ctensor Cc=ctensor::zero(Gdims(3,3,3,3),1);
  
  Cc.view4().add_expand_2(Ac.view3(),Bc.view3());
  print(Cc);

  #endif 


  ctensor D=ctensor::zero(Gdims(3,3));
  
  A.view3().add_contract_aib_aib_ab_to(D.view2(),B.view3());
  print(D);


  #ifdef _WITH_CUDA

  ctensor Dc=ctensor::zero(Gdims(3,3),1);
  Ac.view3().add_contract_aib_aib_ab_to(Dc.view2(),Bc.view3());
  print(Dc);

  #endif
}
