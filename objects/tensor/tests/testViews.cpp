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


int main(int argc, char** argv){

  cnine_session session;

  CtensorB T=CtensorB::sequential({4,4,4});

  cout<<T<<endl; 

  //cout<<T.view3D()<<endl;

  cout<<T.view3D().slice0(0)<<endl;
  cout<<T.view3D().slice0(1)<<endl;
  cout<<T.view3D().slice0(2)<<endl;
  cout<<T.view3D().slice0(3)<<endl;

  cout<<T.view3D().slice1(0)<<endl;
  cout<<T.view3D().slice1(1)<<endl;
  cout<<T.view3D().slice1(2)<<endl;
  cout<<T.view3D().slice1(3)<<endl;

  cout<<T.view3D().slice2(0)<<endl;
  cout<<T.view3D().slice2(1)<<endl;
  cout<<T.view3D().slice2(2)<<endl;
  cout<<T.view3D().slice2(3)<<endl;



}
