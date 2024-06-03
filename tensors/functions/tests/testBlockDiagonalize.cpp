/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "Tensor.hpp"
#include "TensorFunctions.hpp"
#include "CnineSession.hpp"
#include "BlockDiagonalize.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;


  //Tensor<double> A=Tensor<double>::randn({5,5});
  
  Tensor<double> A=oplus(Tensor<double>::random_unitary({3,3}),Tensor<double>::random_unitary({4,4}));
  cout<<A<<endl;


  BlockDiagonalize blocked(A);
  cout<<blocked<<endl;

  //cout<<blocked.U<<endl;
  //cout<<blocked.V<<endl;
  //cout<<blocked.U*transp(blocked.V)<<endl;

  

}

