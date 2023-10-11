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


#ifndef _CnineDimLabels
#define _CnineDimLabels

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{

  class DimLabels{
  public:

    bool _batched=false;
    int _narray=0;


  public: // ---- Copying -----------------------------------------------------------------------------------


    DimLabels copy() const{
      return *this;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    DimLabels& set_batched(const bool x){
      _batched=x;
      return *this;
    }


    int nbatch(const Gdims& dims) const{
      if(_batched) return dims[0];
      else return 0;
    }

    // deprecated 
    Gdims adims(const Gdims& dims) const{
      return dims.chunk(_batched,_narray);
    }

    Gdims gdims(const Gdims& dims) const{
      return dims.chunk(_batched,_narray);
    }

    GstridesB(const GstridesB& x) const{
      return x.chunk(_batched,_narray);
    }

    Gdims ddims(const Gdims& dims) const{
      return dims.chunk(_narray+_batched);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "DimLabels";
    }

    string str(const Gdims& dims) const{
      ostringstream oss;
      
      oss<<"[";
      if(_batched) oss<<"nbatch="<<nbatch(dims)<<",";
      if(_narray>0) oss<<"blocks="<<adims(dims)<<",";
      oss<<"dims="<<ddims(dims)<<" ";
      oss<<"\b]";
      return oss.str();
    }

  };

}

#endif
