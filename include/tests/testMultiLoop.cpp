//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Cnine_base.cpp"

#include "CnineSession.hpp"
#include "MultiLoop.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session(4);

  MultiLoop(10,[](const int i){
      COUT("Starting job  "<<i<<" with "<<nthreads<<" threads.");
      this_thread::sleep_for(chrono::milliseconds(uniform_int_distribution<int>(0,1000)(rndGen))); 
      COUT("Done with job "<<i);
    });

  

}
