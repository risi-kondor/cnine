//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Cnine_base.cpp"

#include "CnineSession.hpp"
#include "Combinations.hpp"

using namespace cnine;

CombinationsBank combinations_bank;


int main(int argc, char** argv){
  cnine_session session(4);

  Combinations C(6,3);
  cout<<C<<endl;


}
