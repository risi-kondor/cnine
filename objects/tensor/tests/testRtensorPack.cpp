/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "RtensorPack.hpp"

using namespace cnine;


int main(int argc, char** argv){
  cnine_session session;

  RtensorPack A0(3,{2,2},fill_sequential());
  cout<<A0<<endl;

  RtensorPack A1(2,{3,3},fill_sequential());
  cout<<A1<<endl;

  RtensorPack B=RtensorPack::cat({A0,A1});
  cout<<B<<endl;

}
