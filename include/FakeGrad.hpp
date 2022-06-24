//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _FakeGrad
#define _FakeGrad

namespace cnine{

  template<typename OBJ> // deprecated 
  class FakeGrad{
  public:

    OBJ* grad=nullptr;

    //~FakeGrad(){
    // if(!is_view) delete grad;
    //}

  public:

    void add_to_grad(const OBJ& x){
      if(grad) grad->add(x);
      else grad=new OBJ(x);
    }

    OBJ& get_grad(){
      if(!grad) grad=new OBJ(OBJ::zeros_like(*this));
      return *grad;
    }

    OBJ view_of_grad(){
      if(!grad) grad=new OBJ(OBJ::zeros_like(*this));
      return grad->_view();
    }

  };

}

#endif 
