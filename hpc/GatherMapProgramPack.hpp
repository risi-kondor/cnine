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

#ifndef _GatherMapProgramPack
#define _GatherMapProgramPack

#include "Cnine_base.hpp"
#include "object_pack.hpp"
#include "GatherMapProgram.hpp"


namespace cnine{


  class GatherMapProgramPack: public object_pack<GatherMapProgram>{
  public:
    
    

  public: // ---- Execution ----------------------------------------------------------------------------------


    template<typename PACK>
    void operator()(const PACK& output, const PACK& arg0){
      const int N=size();
      CNINE_ASSRT(output.size()==N);
      CNINE_ASSRT(arg0.size()==N);

      for(int i=0;  i<N; i++)
	(*this)[i](output.obj[i],arg0.obj[i]);
    }




  };

}


#endif 
