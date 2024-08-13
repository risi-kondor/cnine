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


#ifndef _CnineLtensorEinsumParams
#define _CnineLtensorEinsumParams

#include "Ltensor.hpp"


namespace cnine{

  struct EsumParams{
  public:

    int ddims[4];
    int xstride_d[4];
    int rstride_d[4];

    int sdims[4];
    int xstride_s[4];

    int bdims[4];
    int rstride_b[4];
  };

}

#endif 
