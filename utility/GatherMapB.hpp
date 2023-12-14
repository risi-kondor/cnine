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

#ifndef _GatherMapB
#define _GatherMapB

#include "Cnine_base.hpp"
#include "headed_lists.hpp"

namespace cnine{



  class GatherMapB{
  private:

    headed_lists<int> arr;
    shared_ptr<GatherMapB> _inv;
    int n=0; // why do we need this?
    int* arrg=nullptr; // unsafe!!

  public:

    int in_stride=1;
    int out_stride=1;

    ~GatherMapB(){
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMapB(){}

    GatherMapB(const int _n): 
      n(_n){}

    GatherMapB(const vector<int>& sources, const vector<int>& targets){
      CNINE_ASSRT(sources.size()==targets.size());

      int N=sources.size();
      unordered_map<int,int> sizes;
      for(int i=0; i<N; i++)
	sizes[targets[i]]++;
      
      int n=sizes.size();
      vector<int> heads(n);
      vector<int> lengths(n);
      unordered_map<int,int> mapping;
      int i=0;
      for(auto p:sizes){
	heads[i]=p.first;
	lengths[i]=p.second;
	mapping[p.first]=i;
	i++;
      }

      arr=headed_lists(heads,lengths);
      for(int i=0; i<N; i++){
	arr.push_back(mapping[targets[i]],sources[i]);
      }
    }
    
    
    //GatherMapB(const int _n, const int _nedges, const int _dev=0):
    //arr(n,_nedges,fill::_reserve(),_dev){
    //}


  public: // ---- Conversions --------------------------------------------------------------------------------


    //GatherMapB(const int _n, const array_pool<int>& _arr):
    //arr(_arr), n(_n){
    //}

    //GatherMapB(const int _n, array_pool<int>&& _arr):
    //arr(std::move(_arr)),  n(_n){
    //}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static GatherMapB random(const int _n, const int m, const float p){
      GatherMapB r(_n);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++){
	vector<int> v;
	for(int j=0; j<m; j++)
	  if(distr(rndGen)<p)
	    v.push_back(j);
	r.arr.push_back(i,v);
      }
      return r;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    GatherMapB(const GatherMapB& x, const int _dev):
      arr(x.arr,_dev), n(x.n){
    }

    GatherMapB& move_to_device(const int _dev){
      arr.to_device(_dev);
      return *this;
    }

    int* get_arrg(const int _dev=1){
      if(!arrg) make_arrg();
      return arrg;
    }

    void make_arrg(){
      int memsize=arr.memsize+arr.dir.memsize;
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg, arr.dir.arr, arr.dir.memsize*sizeof(int),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(arrg+arr.dir.memsize, arr.arr, arr.memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      return arr.dev;
    }

    int getn() const{
      return n;
    }

    int size() const{
      return arr.size();
    }

    int offset(const int i) const{
      return arr.offset(i);
    }

    int size_of(const int i) const{
      //return arr.size_of(i)-1;
      return arr.size_of(i);
    }

    int target(const int i) const{
      //return arr(i,0);
      return arr.head(i);
    }

    void set_target(const int i, const int x){
      //arr.set(i,0,x);
      arr.set_head(i,x);
    }

    int operator()(const int i, const int j) const{
      return arr(i,j);
    }

    void set(const int i, const int j, const int x){
      arr.set(i,j,x);
    }

    int push_back(const int len){
      arr.push_back(len);
      return size()-1;
    }

    void for_each(std::function<void(const int i, const int j)> lambda) const{
      int N=size();
      for(int i=0; i<N; i++){
	int M=size_of(i);
	int targt=target(i);
	for(int j=0; j<M; j++)
	  lambda(targt,(*this)(i,j));
      }
    }

    shared_ptr<GatherMapB> inv_ptr() const{
      if(!_inv.get()) make_inv();
      return _inv;
    }

    const GatherMapB& inv() const{
      if(!_inv.get()) make_inv();
      return *_inv;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void make_inv() const{
      map<int,vector<int> > inv_map;
      int total=0;
      for_each([&](const int i, const int j){
	  inv_map[j].push_back(i);
	  total++;
	});
      GatherMapB* r=new GatherMapB(inv_map.rbegin()->first+1);
      r->arr.reserve(size()+total);
      for(auto& p: inv_map)
	r->arr.push_back(p.first,p.second);
      const_cast<GatherMapB&>(*this)._inv.reset(r);
    }

    
    const GatherMapB& sort() const{
      map<int,vector<int> > lengths;
      int N=size();
      for(int i=0; i<N; i++)
	lengths[-size_of(i)].push_back(i);
      GatherMapB r(n);
      r.arr.reserve(arr.tail);
      for(auto& p:lengths){
	int K=-p.first;
	for(auto q:p.second){
	  int i=r.push_back(K);
	  r.set_target(i,target(q));
	  for(int a=0; a<K; a++){
	    r.set(i,a,(*this)(q,a));
	  }
	}
      }
      //for(int i=0; i<r.arr.tail; i++) cout<<r.arr.arr[i]<<" "; cout<<endl;
      //cout<<r.arr.dir<<endl;
      const_cast<GatherMapB&>(*this).arr=std::move(r.arr);
      //cout<<arr.dir<<endl;
      return *this;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GatherMapB";
    }

    string repr() const{
      return "GatherMapB";
    }

    string str(const string indent="") const{
      //for(int i=0; i<arr.tail; i++) cout<<arr.arr[i]<<" "; cout<<endl;
      //cout<<arr.dir<<endl;
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<target(i)<<"<-(";
	for(int j=0; j<size_of(i); j++){
	  oss<<(*this)(i,j)<<",";
	}
	if(size_of(i)>0) oss<<"\b";
	oss<<")\n";
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const GatherMapB& v){
      stream<<v.str(); return stream;}

  };



}

#endif 
