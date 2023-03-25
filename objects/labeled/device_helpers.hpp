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


#ifndef _CnineDeviceHelpers
#define _CnineDeviceHelpers

namespace cnine{

  template<typename TYPE>
  inline void reconcile_devices(TYPE& r, const TYPE& x, std::function<void(TYPE&, const TYPE&)>& lambda){
    if(r.dev==x.dev){
      lambda(r,x);
    }else{
      lambda(r,TYPE(x,r.dev));
    }
  }

  template<typename TYPE>
  inline void reconcile_devices(TYPE& r, const TYPE& x, const TYPE& y, std::function<void(TYPE&, const TYPE&, const TYPE&)> lambda){
    int dev=r.dev;
    if(x.dev==dev){
      if(y.dev==dev)
	lambda(r,x,y);
      else
	lambda(r,x,TYPE(y,dev));
    }else{
      if(y.dev==dev)
	lambda(r,TYPE(x,dev),y);
      else
	lambda(r,TYPE(x,dev),TYPE(y,dev));
    }
  }

}

#endif 
