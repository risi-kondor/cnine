// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef __GindexSet
#define __GindexSet

#include "Cnine_base.hpp"


namespace cnine{


  class GindexSet: public set<int>{
  public:

    using set::set;


  public:

    int first() const{
      return *this->begin();
    }

    int last() const{
      return *this->rbegin();
    }

    bool is_contiguous() const{
      int i=first()-1;
      for(auto p:*this)
	if(p!=(i++)) return false;
      return true;
    }

    bool is_disjoint(const GindexSet& y) const{
      for(auto p:*this)
	if(y.find(p)!=y.end()) return false;
      return true;
    }

    bool covers(const int n) const{
      for(int i=0; i<n; i++)
	if(this->find(i)==this->end()) return false;
      return true;
    }

    bool covers(const int n, const GindexSet& x) const{
      for(int i=0; i<n; i++)
	if((this->find(i)==this->end())&&(x.find(i)==x.end())) return false;
      return true;
    }

    bool covers(const int n, const GindexSet& x, const GindexSet& y) const{
      for(int i=0; i<n; i++)
	if((this->find(i)==this->end())&&(x.find(i)==x.end())&&(x.find(i)==x.end())) return false;
      return true;
    }

    int back() const{
      return *(this->rbegin());
    }


  };

}


#endif 
