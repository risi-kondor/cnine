//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
//#include "CtensorB.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;


  set<int> A={1,3,2};
  set<int> B={2,1,2,3,4};

  for(auto p: A)
    cout<<p<<endl; 

  for(auto p: B)
    cout<<p<<endl; 

  cout<<(A==B)<<endl;

  cout<<*A.rbegin()<<endl;

}
