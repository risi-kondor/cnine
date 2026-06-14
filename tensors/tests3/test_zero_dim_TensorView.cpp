/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "TensorView.hpp"
#include "TensorView_functions.hpp"
#include "CnineSession.hpp"
#include "TensorAccessor.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  Gdims d(0);
  auto A0=TensorView<float>(d,0,0);
  cout<<A0<<endl;

  auto A1=TensorView<float>::init(77);
  cout<<A1<<endl;



}
