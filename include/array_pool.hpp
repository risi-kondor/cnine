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

#ifndef _array_pool
#define _array_pool

#include "Cnine_base.hpp"
#include "IntTensor.hpp"


namespace cnine{

  template<typename TYPE>
  class array_pool{
  public:

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    int memsize=0;
    int tail=0;
    int dev=0;
    bool is_view=false;

    IntTensor dir;

    ~array_pool(){
      if(is_view) return;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    array_pool(): 
      dir(Gdims(0,2),cnine::fill_noalloc()){}


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	TYPE* newarr=new TYPE[newsize];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	TYPE* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(TYPE)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    array_pool(const array_pool& x){
      CNINE_COPY_WARNING();
      dev=x.dev;
      tail=x.tail;
      memsize=tail;
      if(dev==0){
	arr=new TYPE[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_UNIMPL();
      }
      dir=x.dir;
    }

    array_pool(array_pool&& x){
      CNINE_MOVE_WARNING();
      dev=x.dev;
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      dir=std::move(x.dir);
      is_view=x.is_view;
    }

    array_pool<TYPE>& operator=(const array_pool<TYPE>& x){
      CNINE_ASSIGN_WARNING();

      if(is_view){
	arr=nullptr;
	arrg=nullptr;
      }else{
	delete arr; 
	arr=nullptr;
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
	arrg=nullptr;
      }

      dev=x.dev;
      tail=x.tail;
      memsize=x.tail;
      dir=x.dir;
      is_view=false;

      if(dev==0){
	arr=new TYPE[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }


    array_pool<TYPE>& operator=(array_pool<TYPE>&& x){
      CNINE_MOVEASSIGN_WARNING();
      if(!is_view){
	delete arr; 
	arr=nullptr;
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
	arrg=nullptr;
      }
      dev=x.dev;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      tail=x.tail;
      memsize=x.memsize;
      //lookup=x.lookup;
      dir=std::move(x.dir);
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Views --------------------------------------------------------------------------------------


    array_pool<TYPE> view(){
      array_pool<TYPE> R;
      R.dev=dev;
      R.tail=tail;
      R.memsize=memsize;
      R.arr=arr;
      R.arrg=arrg;
      //R.lookup=lookup;
      R.dir=dir;
      R.is_view=true;
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return dir.dim(0);
    }

    int size_of(const int i) const{
      CNINE_ASSRT(i<size());
      return dir(i,1);
    }

    vector<TYPE> operator()(const int i) const{
      CNINE_ASSRT(i<size());
      int addr=dir(i,0);
      int len=dir(i,1);
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }

    void push_back(const vector<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len; i++)
	arr[tail+i]=v[i];
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const set<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      int i=0; 
      for(TYPE p:v){
	arr[tail+i]=p;
	i++;
      }
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const initializer_list<TYPE>& v){
      push_back(vector<TYPE>(v));
    }

    void forall(const std::function<void(const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda((*this)(i));
    }

    void for_each(const std::function<void(const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda((*this)(i));
    }

    vector<vector<TYPE> > as_vecs() const{
      vector<vector<TYPE> > R;
      forall([&](const vector<TYPE>& x){R.push_back(x);});
      return R;
    }

    bool operator==(const array_pool<TYPE>& y) const{
      if(size()!=y.size()) return false;
      for(int i=0; i<size(); i++){
	int n=dir(i,1);
	int offs=dir(i,0);
	int offsy=y.dir(i,0);
	if(n!=y.dir(i,1)) return false;
	for(int j=0; j<n; j++)
	  if(arr[offs+j]!=y.arr[offsy+j]) return false;
      }
      return true;
    }


  public: // ---- Specialized --------------------------------------------------------------------------------

    /*
    vector<TYPE> subarray_of(const int i, const int beg) const{
      assert(i<size());
      auto& p=lookup[i];
      int addr=p.first+beg;
      int len=p.second-beg;
      assert(len>=0);
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }


    void push_back_cat(TYPE first, const vector<TYPE>& v){
      int len=v.size()+1;
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      arr[tail]=first;
      for(int i=0; i<len-1; i++)
	arr[tail+1+i]=v[i];
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "array_pool";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	auto v=(*this)(i);
	oss<<"(";
	for(int j=0; j<v.size()-1; j++)
	  oss<<v[j]<<",";
	if(v.size()>0) oss<<v.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const array_pool& v){
      stream<<v.str(); return stream;}

  };

}

#endif
