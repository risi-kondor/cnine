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

#ifndef _compact_int_tree
#define _compact_int_tree

#include "Cnine_base.hpp"
#include "auto_array.hpp"


namespace cnine{

      
  class int_tree: public auto_array<int>{
  public:

    class node{
    public:

      int_tree* owner;
      int ptr;

      node(int_tree* _owner, const int _ptr):
	owner(_owner), ptr(_ptr){}


    public: // ---- Access ------------------------------------------------


      int label() const{
	return (*owner)[ptr];
      }

      int nchildren() const{
	return (*owner)[ptr+1];
      }
	
      int parent() const{
	return (*owner)[ptr+2];
      }
	
      node child(const int i) const{
	CNINE_ASSRT(i<nchildren());
	return owner->node_at(ptr+i+3);
      }

      node add_child(const int i, const int v, const int n){
	return owner->add_node(ptr,i,v,n);
      }

      void for_each_child(std::function<void(node)>& lambda){
	int n=nchildren();
	for(int i=0; i<n; i++)
	  lambda(child(i));
      }
      
    public: // ---- I/O --------------------------------------------------

      string print_recursively(){
	ostringstream oss;
	oss<<"("<<label();
	int n=nchildren();
	if(n>0){
	  oss<<",";
	  for(int i=0; i<n-1; i++)
	    oss<<child(i).print_recursively()<<",";
	  oss<<child(n-1).print_recursively();
	}
	oss<<")";
	return oss.str();
      }

    };
    

  public: //---- Constructors --------------------------------


    int_tree(){}


  public: //---- Copying -------------------------------------


    int_tree(const int_tree& x)=delete;


  public: //---- Access -------------------------------------


    node root(){
      return node(this,0);
    }

    node node_at(const int p){
      return node(this,p);
    }

    node add_root(const int v, const int n){
      CNINE_ASSRT(size()==0);
      resize(n+3);
      set(0,v);
      set(1,n);
      set(2,-1);
      for(int i=0; i<n; i++)
	set(i+3,-1);
      return node_at(0);
    }

    node add_node(const int parent, const int i, const int v, const int n){
      CNINE_ASSRT(i<get(parent+1));
      CNINE_ASSRT(get(parent+i+3)==-1);
      int tail=_size;
      set(parent+i+3,tail);
      resize(tail+n+3);
      
      set(tail,v);
      set(tail+1,n);
      set(tail+2,parent);
      for(int i=0; i<n; i++)
	set(tail+i+3,-1);

      return node_at(tail);
    }

    void traverse(const std::function<void(const node&)>& lambda, const int p=0){
      lambda(node_at(p));
      int m=get(p+1);
      for(int i=0; i<m; i++)
	lambda(node_at(p+3+i));
    }


  public: //---- Operations  --------------------------------------


    vector<int> depth_first_traversal() const{
      vector<int> r;
      depth_first_traversal(0,r);
      return r; 
    }

    void depth_first_traversal(const int p, vector<int>& r) const{
      r.push_back(get(p));
      int m=get(p+1);
      for(int i=0; i<m; i++)
	depth_first_traversal(p+3+i,r);
    }

    /*
    vector<int> semi_depth_first_traversal() const{
      vector<int> r;
      r.push_back(0)
      semi_depth_first_traversal(0,r);
      return r; 
    }

    void semi_depth_first_traversal(const int p, const vector<int>& r){
      int m=get(p+1);
      for(int i=0; i<m; i++)
	r.push_back(p+3+i);
      for(int i=0; i<m; i++)
	semi_depth_first_traversal(p+3+i);
    }
    */


  public: //---- I/O ----------------------------------------------


    string str(){
      ostringstream oss;
      oss<<root().print_recursively();
      return oss.str();
    }

  };

}


#endif 
