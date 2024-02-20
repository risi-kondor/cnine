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


#ifndef _CnineMemoryManager
#define _CnineMemoryManager

#include "Cnine_base.hpp"

namespace cnine{

  class MemoryManager{
  public:

    virtual ~MemoryManager()=0;
    virtual void* malloc(const int n) const=0;
    virtual void free(void* p) const=0;
    virtual void clear() const=0;

  };


}

#endif 
