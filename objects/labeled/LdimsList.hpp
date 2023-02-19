/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef __LdimsList
#define __LdimsList

#include "Cnine_base.hpp"
#include "GindexSet.hpp"
#include "Ldims.hpp"


namespace cnine{


  class LdimsList: public vector<Ldims*>{
  public:

    ~LdimsList(){
      for(auto p:*this)
	delete p;
    }

    LdimsList(const initializer_list<Ldims>& _ldims){
      for(auto& p:_ldims)
	push_back(p.clone());
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)[i]->str();
	if(i<size()-1) oss<<",";
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LdimsList& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
