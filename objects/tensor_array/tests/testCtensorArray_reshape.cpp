//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CtensorArray.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;
typedef CtensorArray ctensora;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ctensor T=ctensor::sequential(dims(4,4));


  ctensora A=ctensora::sequential(dims(2,2),dims(4,4));
  printl("A",A);

  ctensora B=A.view_as_shape(dims(4));
  printl("B",B);

}
