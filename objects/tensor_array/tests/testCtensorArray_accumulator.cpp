//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CtensorArray_funs.hpp"

#include "CtensorA_plus_cop.hpp"
#include "CtensorA_add_Mprod_cop.hpp"

#include "CellMask2r.hpp"
#include "accumulate_cmap.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;

typedef CtensorA_plus_cop CtensorA_plus;


int main(int argc, char** argv){
  cnine_session genet;
  device dev=deviceid::GPU0;
  cout<<endl;

  CtensorArray A(dims(4),dims(2,2),[](const Gindex& aix, const Gindex& cix){return aix(0);});
  CtensorArray B(dims(4),dims(2,2),[](const Gindex& aix, const Gindex& cix){
      return complex<float>(0,aix(0));});
  CtensorArray C(dims(4),dims(2,2),fill::zero);

  CellMask2r mask(dims(4),dims(4),dims(4));
  mask.push(Gindex(0),Gindex(1),Gindex(2));
  mask.push(Gindex(0),Gindex(1),Gindex(3));
  mask.push(Gindex(1),Gindex(3),Gindex(3));
  cout<<mask.str()<<endl;

  //add_accumulate<CtensorA_plus>(mask,C,A,B);
  //print(C);

#ifdef _WITH_CUDA 

  CtensorArray Ag=A.to(dev);
  CtensorArray Bg=B.to(dev);

  CtensorArray Cg(dims(4),dims(2,2),fill::zero,dev);
  add_accumulate<CtensorA_plus>(mask,Cg,Ag,Bg);
  printl("add_accumulate<CtensorA_plus>(mask,Cg,Ag,Bg)",Cg);

#endif 

}

