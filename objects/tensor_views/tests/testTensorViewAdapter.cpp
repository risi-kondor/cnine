//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Cnine_base.cpp"

#include "CnineSession.hpp"
#include "RtensorA.hpp"

#include "GindexFuser.hpp"
#include "GindexSymm.hpp"
#include "IndexingAdapter.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session(4);

  //GindexFuser fuse0(3,3);
  GindexSymm fuse0(3,3);
  cout<<fuse0(2,2)<<endl;

  RtensorA A({9},fill::sequential);
  cout<<A<<endl;


  IndexingAdapter1<RtensorA,GindexSymm> B(A,fuse0);
  //cout<<B(0,2)<<endl;
  cout<<B.to<RtensorA>()<<endl;

}
