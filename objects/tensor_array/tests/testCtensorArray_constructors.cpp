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


  ctensora A=ctensora::zero(dims(2,2),dims(4,4));
  printl("A",A);

  ctensora B=ctensora::sequential(dims(2,2),dims(4,4));
  printl("B",B);

  ctensora C=ctensora::gaussian(dims(2,2),dims(4,4));
  printl("C",C);

  
  //ctensora T1(T);
  //printl("T1",T1);

  //ctensora T2(dims(2,2),T);
  //printl("T2",T2);


  ctensora L1(dims(2,2),dims(4,4),[](const Gindex& aix, const Gindex& ix){
      return complex<float>(aix(dims(2,2)),ix(dims(4,4)));});
  printl("L1",L1);
    
  //ctensora V1(B.adims,B.cdims,fill::view,B.arr,B.arrc);
  //printl("V1",V1);

}
