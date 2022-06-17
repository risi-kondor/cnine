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

#include "CtensorA_copy_cop.hpp"

#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "CellwiseBinaryCmap.hpp"
#include "MVprodCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;

typedef CtensorA_copy_cop Ctensor_copy;


int main(int argc, char** argv){
  cnine_session genet;
  cout<<endl;

  CtensorArray A(dims(2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return aix(0);});

  //printl("cellwise",cellwise<Ctensor_copy>(A));

}
