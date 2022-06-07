//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorB.hpp"
#include "CtensorB_array.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef CtensorB ctensor; 
typedef CtensorB_array ctensor_arr; 


int main(int argc, char** argv){

  cnine_session session;

  Gdims adims({2,2});

  ctensor_arr A=ctensor_arr::sequential(adims,{4,4});
  cout<<"A="<<endl<<A<<endl<<endl;;

  ctensor_arr B=ctensor_arr::gaussian(adims,{4,4});
  cout<<"B="<<endl<<B<<endl<<endl;

}
