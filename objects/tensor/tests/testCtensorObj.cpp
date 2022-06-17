//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef CscalarObj cscalar; 
typedef CtensorObj ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensor A=ctensor::sequential({4,4});
  cout<<"A="<<endl<<A<<endl<<endl;;

  ctensor B=ctensor::gaussian({4,4});
  cout<<"B="<<endl<<B<<endl<<endl;

  cscalar c(2.0);

  cout<<"A+B="<<endl<<A+B<<endl<<endl;
  cout<<"A-B="<<endl<<A-B<<endl<<endl;
  cout<<"c*A="<<endl<<c*A<<endl<<endl;
  cout<<"A*B="<<endl<<A*B<<endl<<endl;
  cout<<endl; 

  cout<<"norm2(A)="<<endl<<norm2(A)<<endl<<endl;
  cout<<"inp(A,B)="<<endl<<inp(A,B)<<endl<<endl;
  cout<<"ReLU(A,0.1)="<<endl<<ReLU(A,0.1)<<endl<<endl;
  cout<<endl; 

  ctensor C=ctensor::zero({4,4});
  C+=A;
  C+=A;
  C+=A;
  C+=A;
  cout<<"A+A+A+A="<<endl<<C<<endl;
  cout<<endl; 

  ctensor N=A.col_norms();
  cout<<N<<endl; 

  ctensor D=A.divide_cols(N);
  cout<<D<<endl;
  cout<<D.col_norms()<<endl; 

  //cout<<"  fn(C) = "<<endl<<C.apply([](const complex<float> x){return x*x+complex<float>(3.0);})<<endl; 
  //cout<<"  fn2(C) = "<<endl<<C.apply([](const int i, const int j, const complex<float> x){return x+complex<float>(i,j);})<<endl; 

  cout<<ctensor(Gdims({5,5}),[](const int i, const int j){return complex<float>(i+j);})<<endl;

  cout<<endl;
}
