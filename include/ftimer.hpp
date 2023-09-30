/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineFtimer
#define _CnineFtimer

#include <fstream>
#include <chrono>
#include <ctime>
#include "CnineLog.hpp"

extern cnine::CnineLog cnine::cnine_log;

namespace cnine{

  class ftimer{
  public:

    string name;
    chrono::time_point<chrono::system_clock> t0;

    ftimer(const string _name):
      name(_name), t0(chrono::system_clock::now()){}

    ~ftimer(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      cnine::cnine_log.log_call(name,elapsed);
    }

  };

}


#endif 
