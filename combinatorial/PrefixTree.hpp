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

#ifndef _PrefixTree
#define _PrefixTree

#include "Cnine_base.hpp"
#include <unordered_map>


namespace cnine{

  template<typename TYPE>
  class PrefixTree{
  public:

    unordered_map<TYPE,PrefixTree*> children;

    ~PrefixTree(){
      for(auto& p:children)
	delete p.second;
    }


    PrefixTree(){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    PrefixTree(const PrefixTree& x){
      for(auto& p: x.children)
	children[p.first]=new PrefixTree(*p.second);
    }

    PrefixTree(PrefixTree&& x):
      children(std::move(x.children)){
      x.children.clear();
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    PrefixTree* branch(const TYPE& x) const{
      auto it=children.find(x);
      if(it!=children.end()) return it->second;
      else return nullptr;
    }

    PrefixTree& get_branch(const TYPE& x){
      auto it=children.find(x);
      if(it!=children.end()) return *(it->second);
      children[x]=new PrefixTree();
      return *children[x];
    }

    bool find(const TYPE& x) const{
      return branch(x)!=nullptr;
    }

    bool find(const vector<TYPE>& x) const{
      if(x.size()==0) return true;
      auto p=branch(x[0]);
      if(!p) return false;
      p->find(vector<TYPE>(x.begin()+1,x.end()));
    }

    void add_path(const vector<TYPE> x){
      if(x.size()==0) return;
      get_branch(x[0]).add_path(vector<TYPE>(x.begin()+1,x.end()));
    }

    void forall_maximal_paths(const std::function<void(const vector<TYPE>&)> lambda) const{
      vector<TYPE> prefix;
      forall_maximal_paths(prefix,lambda);
    }

    void forall_maximal_paths(vector<TYPE>& prefix, const std::function<void(const vector<TYPE>&)> lambda) const{
      if(children.size()==0) lambda(prefix);
      for(auto& p: children){
	prefix.push_back(p.first);
	p.second->forall_maximal_paths(prefix,lambda);
	prefix.pop_back();
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      forall_maximal_paths([&](const vector<TYPE>& x){
	  oss<<indent<<"(";
	  for(int i=0; i<x.size()-1; i++) oss<<x[i]<<",";
	  if(x.size()>0) oss<<x.back();
	  oss<<")"<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const PrefixTree& x){
      stream<<x.str(); return stream;}

  };

    

}

#endif 
