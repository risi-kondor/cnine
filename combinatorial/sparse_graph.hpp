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

#ifndef _CnineSparseGraph
#define _CnineSparseGraph

#include "Cnine_base.hpp"
#include <unordered_map>
#include "map_of_maps.hpp"
#include "labeled_tree.hpp"


namespace cnine{

  // This class represents a weighted, undirected graph

  template<typename KEY, typename TYPE, typename LABEL=int>
  class sparse_graph: public map_of_maps<KEY,KEY,TYPE>{
  public:

    typedef map_of_maps<KEY,KEY,TYPE> BASE;

    using BASE::data;
    using BASE::operator==;

    int n=0;

    Tensor<LABEL> labels;
    bool labeled=false;


  public: // ---- Constructors -------------------------------------------------------------------------------


    sparse_graph(){};

    sparse_graph(const int _n):
      n(_n){}

    sparse_graph(const initializer_list<pair<KEY,KEY> >& list):
      sparse_graph([](const initializer_list<pair<KEY,KEY> >& list){
	  TYPE t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t);
	  return t+1;}(list), list){}

    sparse_graph(const int _n, const initializer_list<pair<KEY,KEY> >& list): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    sparse_graph(const vector<pair<KEY,KEY> >& list):
      sparse_graph([](const vector<pair<KEY,KEY> >& list){
	  KEY t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t);
	  return t+1;}(list), list){}

    sparse_graph(const int _n, const vector<pair<KEY,KEY> >& list): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    sparse_graph(const int _n, const vector<pair<KEY,KEY> >& list, const TensorView<LABEL>& L): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
      labels=L;
    }


    sparse_graph(const int n, const TensorView<int>& _edges):
      sparse_graph(n){
      CNINE_ASSRT(_edges.ndims()==2);
      CNINE_ASSRT(_edges.get_dim(0)==2);
      CNINE_ASSRT(_edges.max()<n);
      int nedges=_edges.get_dim(1);
      for(int i=0; i<nedges; i++)
	set(_edges(0,i),_edges(1,i),1.0);
    }


  public: // ---- Named Constructors -------------------------------------------------------------------------


    static sparse_graph trivial(){
      return sparse_graph(1,{});}

    static sparse_graph edge(){
      return sparse_graph(2,{{0,1}});}

    static sparse_graph triangle(){
      return sparse_graph(3,{{0,1},{1,2},{2,0}});}

    static sparse_graph square(){
      return sparse_graph(4,{{0,1},{1,2},{2,3},{3,0}});}

    static sparse_graph complete(const int n){
      vector<pair<int,int> > v;
      for(int i=0; i<n; i++)
	for(int j=0; j<i; j++)
	  v.push_back(pair<int,int>(i,j));
      return sparse_graph(n,v);
    }

    static sparse_graph cycle(const int n){
      vector<pair<int,int> > v;
      for(int i=0; i<n-1; i++)
	v.push_back(pair<int,int>(i,i+1));
      v.push_back(pair<int,int>(n-1,0));
      return sparse_graph(n,v);
    }

    static sparse_graph star(const int m){
      vector<pair<int,int> > v(m);
      for(int i=0; i<m; i++)
	v[i]=pair<int,int>(0,i+1);
      return sparse_graph(m+1,v);
    }

    static sparse_graph random(const int _n, const float p=0.5){
      sparse_graph G(_n);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++) 
	for(int j=0; j<i; j++)
	  if(distr(rndGen)<p){
	    G.set(i,j,1.0);
	    G.set(j,i,1.0);
	  }
      return G;
    }


  public: // ---- Conversions ------------------------------------------------------------------------------


    sparse_graph(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(x.dims(0)==x.dims(1));
      n=x.dim(0);
      for(int i=0; i<n; i++)
	for(int j=0; j<=i; j++)
	  if(x(i,j)) set(i,j,x(i,j));
    }

    Tensor<TYPE> dense() const{
      auto R=Tensor<TYPE>::zero({n,n});
      BASE::for_each([&](const KEY& i, const KEY& j, const TYPE& v){
	  R.set(i,j,v);});
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    int nedges() const{
      return BASE::size()/2;
    }

    bool is_labeled() const{
      return labeled;
    }

    bool is_neighbor(const KEY& i, const KEY& j) const{
      return is_filled(i,j);
    }
    
    void set(const KEY& i, const KEY& j, const TYPE& v){
      BASE::set(i,j,v);
      BASE::set(j,i,v);
    }

    vector<int> neighbors(const int i) const{
      vector<int> r;
      const auto _r=data[i];
      for(auto& p: _r)
	r.push_back(p.first);
      return r;
    }

    template<typename TYPE2>
    void insert(const sparse_graph<KEY,TYPE2>& H, const vector<int>& v){
      for(auto p:v)
	CNINE_ASSRT(p<n);
      H.for_each_edge([&](const int i, const int j, const TYPE2& val){
	  set(v[i],v[j],val);});
    }


  public: // ---- Lambdas ----------------------------------------------------------------------------------


    void for_each_edge(std::function<void(const int, const int)> lambda, const bool self=0) const{
      BASE::for_each([&](const int i, const int j, const TYPE& v){if(i<=j) lambda(i,j);});
    }

    void for_each_edge(std::function<void(const int, const int, const TYPE&)> lambda, const bool self=0) const{
      BASE::for_each([&](const int i, const int j, const TYPE& v){if(i<=j) lambda(i,j,v);});
    }


  public: // ---- Subgraphs ----------------------------------------------------------------------------------


    labeled_tree<KEY> greedy_spanning_tree(const KEY root=0) const{
      //CNINE_ASSRT(root<n);
      labeled_tree<KEY> r(root);
      vector<bool> matched(n,false);
      matched[root]=true;
      for(auto& p: BASE::data[root]){
	if(!p.second) continue;
	matched[p.first]=true;
	r.children.push_back(greedy_spanning_tree(p.first,matched));
      }
      return r;
    }


  private:


    labeled_tree<KEY>* greedy_spanning_tree(const int v, vector<bool>& matched) const{
      labeled_tree<KEY>* r=new labeled_tree<KEY>(v);
      for(auto& p: BASE::data[v]){
	if(!p.second) continue;
	if(matched[p.first]) continue;
	matched[p.first]=true;
	r->children.push_back(greedy_spanning_tree(p.first,matched));
      }
      return r;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "sparse_graph";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Graph with "<<n<<" vertices:"<<endl;
      oss<<dense().str(indent+"  ")<<endl;
      //if(is_labeled) oss<<labels.str(indent+"  ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const sparse_graph& x){
      stream<<x.str(); return stream;}


  };

}


namespace std{

  template<typename KEY, typename TYPE, typename LABEL>
  struct hash<cnine::sparse_graph<KEY,TYPE,LABEL> >{
  public:
    size_t operator()(const cnine::sparse_graph<KEY,TYPE,LABEL>& x) const{
      return 1;
      //      if(x.is_labeled()) return (hash<cnine::SparseRmatrix>()(x)<<1)^hash<cnine::RtensorA>()(x.labels);
      //return hash<cnine::SparseRmatrix>()(x);
    }
  };
}




#endif 
