// This file is part of cnine, a symmetric group FFT library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CombinationsBank
#define _CombinationsBank

#include "Cnine_base.hpp"
#include "CombinationsB.hpp"


namespace cnine{


  class CombinationsBranch: public unordered_map<int,CombinationsB*>{
  public:

    const int n;

    CombinationsBranch(const int _n): n(_n){}

    ~CombinationsBranch(){
      for(auto& p:*this)
	delete p.second;
    }

  public:

    //CombinationsB get(const int m){
      
    //}

  };



  class CombinationsBank{
  public:

    unordered_map<int,CombinationsBranch*> branches;

    ~CombinationsBank(){
      for(auto& p:branches)
	delete p.second;
    }


  public:

    CombinationsBranch& branch(const int n){
      auto it=branches.find(n);
      if(it!=branches.end()) return *it->second;
      branches[n]=new CombinationsBranch(n);
      return *branches[n];
    }


    CombinationsB* get(const int n, const int m){
      CombinationsBranch& br=branch(n);
      auto it=br.find(m);
      if(it!=br.end()) return it->second;
      
      CombinationsB* C=new CombinationsB(n,m);
      if(m==1){
	C->N=n;
      }else{
	for(int i=0; i<n-m+1; i++){
	  CombinationsB* sub=get(n-i-1,m-1);
	  C->N+=sub->n;
	  C->sub.push_back(sub);
	}
      }

      br[m]=C;
      return C;
    }


  };

}

#endif 
