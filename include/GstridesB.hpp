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


#ifndef __GstridesB
#define __GstridesB

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"

namespace cnine{


  class TensorPackDir;


  class GstridesB: public vector<std::size_t>{
  public:

    friend class TensorPackDir;

    typedef std::size_t size_t;

    GstridesB(){}

    GstridesB(const int k, const fill_raw& dummy): 
      vector<size_t>(k){}

    GstridesB(const int k, const fill_zero& dummy): 
      vector<size_t>(k,0){}

    GstridesB(const initializer_list<size_t>& lst):
      vector<size_t>(lst){}

    GstridesB(const initializer_list<int>& lst):
      vector<size_t>(lst.size()){
      for(auto p:lst)
	push_back(p);
    }

    GstridesB(const int i0): vector<size_t>(1){
      (*this)[0]=i0;
    }

    GstridesB(const int i0, const int i1): vector<size_t>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    GstridesB(const int i0, const int i1, const int i2): vector<size_t>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3): vector<size_t>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3, const int i4): vector<size_t>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }


    GstridesB(const Gdims& dims, const int s0=1): 
      vector<size_t>(dims.size()){
      int k=dims.size();
      assert(k>0);
      (*this)[k-1]=s0;
      for(int i=k-2; i>=0; i--)
      (*this)[i]=(*this)[i+1]*dims[i+1];
      //regular=true;
    }

    GstridesB(const vector<int>& x):
      vector<size_t>(x.size()){
      for(int i=0; i<x.size(); i++)
	(*this)[i]=x[i];
    }


#ifdef _WITH_ATEN
    GstridesB(const at::Tensor& T):
      GstridesB(T.dim(),fill_raw()){
      for(int i=0; i<size() ; i++)
	(*this)[i]=T.stride(i);
    }
#endif 


  public:

    size_t operator()(const int i) const{
      if(i<0) return (*this)[size()+i];
      return (*this)[i];
    }

    size_t back(const int i=0) const{
      return (*this)[size()-1-i];
    }

    bool is_regular(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      size_t k=size();
      int t=1;
      for(int i=k-1; i>=0; i--){
	if((*this)[i]!=t) return false;
	t*=dims[i];
      }
      return true;
    }

    bool is_contiguous(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      if(is_regular(dims)) return true;

      vector<size_t> v(*this);
      int nz=0; 
      for(int i=0; i<size(); i++) 
	if(v[i]>0) nz++;

      size_t t=1;
      for(int i=0; i<nz; i++){
	auto it=std::find(v.begin(),v.end(),t);
	if(it==v.end()) return false;
	int a=it-v.begin();
	v[a]=0;
	t*=dims[a];
      }

      return true;
    }

    size_t memsize(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      if(dims.asize()==0) return 0;
      size_t t=0;
      for(int i=0; i<size(); i++)
	t=std::max(t,(*this)[i]*dims[i]);
      return t;
    }

    Gstrides reals() const{
      Gstrides R(size(),fill_raw());
      for(int i=0; i<size(); i++)
	R[i]=(*this)[i]*2;
      return R;
    }

    [[deprecated]]
    size_t total() const{
      size_t t=1; 
      for(int i=0; i<size(); i++) t*=(*this)[i];
      return t;
    }

    bool operator==(const GstridesB& x) const{
      if(size()!=x.size()) return false;
      for(int i=0; i<size(); i++)
	if((*this)[i]!=x[i]) return false;
      return true;
    }


  public: // ---- Indexing -----------------------------------------------------------------------------------


    size_t operator()(const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size());
      size_t t=0; //offset;
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i]*ix[i];
      return t;
    }

    size_t offs(const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size());
      size_t t=0;
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i]*ix[i];
      return t;
    }

    size_t offs(const int i, const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size()-1);
      size_t t=((*this)[0]);
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i+1]*ix[i];
      return t;
    }

    size_t offs(const int i0) const{
      return i0*(*this)[0];
    }

    size_t offs(const int i0, const int i1) const{
      return i0*(*this)[0]+i1*(*this)[1];
    }

    size_t offs(const int i0, const int i1, const int i2) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2];
    }


    size_t offs(const int i0, const int i1, const int i2, const int i3) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3];
    }

    size_t offs(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3]+i4*(*this)[4];
    }

    size_t combine(const vector<int>& v) const{
      size_t t=0;
      for(auto p:v){
	assert(p<size());
	t+=(*this)[p];
      }
      return t;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    GstridesB transp() const{
      int len=size();
      CNINE_ASSRT(len>=2);
      if(len==2) return GstridesB((*this)[1],(*this)[0]);
      GstridesB r(*this);
      std::swap(r[len-2],r[len-1]);
      return r;
    }

    GstridesB permute(const vector<int>& p) const{
      CNINE_ASSRT(p.size()<=size());
      GstridesB R;
      R.resize(size());
      for(int i=0; i<p.size(); i++)
	R[i]=(*this)[p[i]];
      for(int i=p.size(); i<size(); i++)
	R[i]=(*this)[i];
      return R;
    }

    GstridesB fuse(const int a, const int n) const{
      GstridesB R(size()-n+1,fill_raw());
      for(int i=0; i<a; i++) R[i]=(*this)[i];
      for(int i=0; i<size()-(a+n-1); i++) R[a+i]=(*this)[a+n+i-1];
      return R;//.set_offset(offset);
    }
    
    GstridesB remove(const int j) const{
      GstridesB R;
      assert(j<size());
      if(size()==1){
	R.push_back(1);
	return R;
      }
      if(j<0){
	for(int i=0; i<size(); i++)
	  if(i!=size()+j) R.push_back((*this)[i]);
      }else{
	for(int i=0; i<size(); i++)
	  if(i!=j) R.push_back((*this)[i]);
      }
      return R;//.set_offset(offset);
    }
    

    GstridesB insert(const int d, const int x) const{
      GstridesB r(size()+1,fill_raw());
      for(int i=0; i<d; i++) r[i]=(*this)[i];
      r[d]=x;
      for(int i=d; i<size(); i++) r[i+1]=(*this)[i];
      return r;
    }

    GstridesB append(const int s) const{
      GstridesB R(*this);
      R.push_back(s);
      return R;//.set_offset(offset);
    }

    GstridesB cat(const GstridesB& y) const{
      GstridesB R(size()+y.size(),fill_raw());
      for(int i=0; i<size(); i++) R[i]=(*this)[i];
      for(int i=0; i<y.size(); i++) R[size()+i]=y[i];
      return R;
    }

    GstridesB prepend(const int i) const{
      if(i<0) return *this;
      GstridesB R;
      R.push_back(i);
      for(auto p:*this) R.push_back(p);
      return R;
    }

    GstridesB chunk(int beg, int n=-1) const{
      if(beg<0) beg=size()+beg;
      if(n==-1) n=size()-beg;
      GstridesB R(n,fill_raw());
      for(int i=0; i<n; i++)
	R[i]=(*this)[beg+i];
      return R;//.set_offset(offset);
    }

    size_t offs(int j, const GstridesB& source) const{
      assert(source.size()==size());
      size_t t=0;
      for(int i=0; i<size(); i++){
	size_t r=j/source[i];
	t+=(*this)[i]*r;
	j-=r*source[i];
      }
      return t;
    }


  public: // ---- offset -------------------------------------------------------------------------------------
    

    // deprecated
    //GstridesB& set_offset(const int i){
    //offset=i;
    //return *this;
    //}

    // deprecated 
    //GstridesB& inc_offset(const int i){
      //offset+=i;
      //return *this;
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      int k=size();
      oss<<indent<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<")";//["<<offset<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GstridesB& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 
