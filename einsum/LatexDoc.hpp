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


#ifndef _LatexDoc
#define _LatexDoc

#include "TensorView.hpp"

namespace cnine{


  class LatexDoc: public string{
  public:

    LatexDoc(){
      (*this)+="\\documentclass{article}\n";
      (*this)+="\\begin{document}\n";
    }

    LatexDoc(const string str):
      LatexDoc(){
      (*this)+=str;
      finish();
    }

    void finish(){
      (*this)+="\n \\end{document}\n";
    }

    


  };

}

#endif 
