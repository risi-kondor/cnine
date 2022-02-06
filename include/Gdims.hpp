// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef __Gdims
#define __Gdims

#include "Cnine_base.hpp"
#include "GindexSet.hpp"
//#include "Bifstream.hpp"
//#include "Bofstream.hpp"


namespace cnine{


  class Gdims: public vector<int>{
  public:

    Gdims(){}

    //Gdims(const vector<int>& x): vector<int>(x){}

    Gdims(const int k, const fill_raw& dummy): 
      vector<int>(k){}

    Gdims(const vector<int>& x){
      for(auto p:x) if(p>=0) push_back(p);
    }

    Gdims(const initializer_list<int>& x){
      for(auto p:x) if(p>=0) push_back(p);
    }

    Gdims(const int i0): vector<int>(1){
      (*this)[0]=i0;
    }

    Gdims(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    Gdims(const int i0, const int i1, const int i2): vector<int>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    Gdims(const int i0, const int i1, const int i2, const int i3): vector<int>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    Gdims(const int i0, const int i1, const int i2, const int i3, const int i4): vector<int>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }

    Gdims(const Gdims& d1, const Gdims& d2): vector<int>(d1.size()+d2.size()){
      for(int i=0; i<d1.size(); i++) (*this)[i]=d1[i];
      for(int i=0; i<d2.size(); i++) (*this)[i+d1.size()]=d2[i];
    }


  public:

    int k() const{
      return size();
    }

    int operator()(const int i) const{
      return (*this)[i];
    }

    void set(const int i, const int x){
      (*this)[i]=x;
    }

    int asize() const{
      int t=1; for(int i=0; i<size(); i++) t*=(*this)[i];
      return t;
    }

    int first() const{
      return (*this)[0];
    }

    int last() const{
      return (*this)[size()-1];
    }

    bool operator==(const Gdims& x) const{
      if(size()!=x.size()) return false;
      for(int i=0; i<size(); i++)
	if((*this)[i]!=x[i]) return false;
      return true;
    }


  public:

    int combined(const int a, const int b) const{
      assert(a<=b);
      assert(b<=size());
      int t=1; for(int i=a; i<b; i++) t*=(*this)[i];
      return t;
    }

    Gdims chunk(const int beg, int n=-1) const{
      if(n==-1) n=size()-beg;
      Gdims R;
      for(int i=0; i<n; i++)
	R.push_back((*this)[beg+i]);
      return R;
    }

    Gdims remove(const int j) const{
      Gdims R;
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
      return R;
    }

    Gdims insert(const int j, const int n) const{
      Gdims R;
      for(int i=0; i<j; i++) R.push_back((*this)[i]);
      R.push_back(n);
      for(int i=j; i<size(); i++) R.push_back((*this)[i]);
      return R;
    }


    Gdims append(const int i) const{
      Gdims R=*this;
      if(i>=0) R.push_back(i);
      return R;
    }

    Gdims cat(const Gdims& y) const{
      Gdims R(size()+y.size(),fill_raw());
      for(int i=0; i<size(); i++) R[i]=(*this)[i];
      for(int i=0; i<y.size(); i++) R[size()+i]=y[i];
      return R;
    }

     Gdims prepend(const int i) const{
      if(i<0) return *this;
      Gdims R;
      R.push_back(i);
      for(auto p:*this) R.push_back(p);
      return R;
    }

    Gdims transpose() const{
      assert(size()==2);
      return Gdims((*this)[1],(*this)[0]);
    }

    Gdims convolve(const Gdims& y) const{
      assert(size()==y.size());
      Gdims R(*this);
      for(int i=0; i<size(); i++)
	R[i]-=y[i]-1;
      return R;
    }

    Gdims Mprod(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

    Gdims Mprod_AT(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
    }

   Gdims Mprod_TA(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

    Gdims Mprod_TT(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
    }


  public:

    Gdims select(const GindexSet& s) const{
      Gdims r;
      for(auto p: s){
	assert(p<size());
	r.push_back((*this)[p]);
      }
      return r;
    }

    int unite(const GindexSet& s) const{
      int r=1;
      for(auto p: s){
	assert(p<size());
	r*=(*this)[p];
      }
      return r;
    }


  public:
    
    vector<int> strides() const{
      int k=size();
      vector<int> R(k);
      if(k==0) return R;
      R[k-1]=1;
      for(int i=k-2; i>=0; i--)
	R[i]=(*this)[i+1]*R[i+1];
      return R;
    }

    template<typename TYPE>
    vector<TYPE> to_vec() const{
      vector<TYPE> R(size());
      for(int i=0; i<size(); i++)
	R[i]=(*this)[i];
      return R;
    }


  public: // checks

    void check_eq(const Gdims& x) const{
      if(!((*this)==x)) throw std::out_of_range("Tensor dimensions "+str()+" do not match "+x.str()+".");
    }

    void check_cell_eq(const Gdims& x) const{
      if(!((*this)==x)) throw std::out_of_range("Tensor cell dimensions "+str()+" do not match "+x.str()+".");
    }


  public:

    /*
    Gdims(Bifstream& ifs){
      int _k=ifs.get<int>();
      resize(_k);
      for(int i=0; i<_k; i++)
	(*this)[i]=ifs.get<int>();
    }

    void serialize(Bofstream& ofs) const{
      const int k=size();
      ofs.write(k);
      for(int i=0; i<k; i++)
	ofs.write((*this)[i]);
    }
    */

    string str() const{
      ostringstream oss;
      int k=size();
      oss<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    string repr() const{
      return "<cnine::Gdims"+str()+">";
    }

    friend ostream& operator<<(ostream& stream, const Gdims& x){
      stream<<x.str(); return stream;
    }


  };

  inline Gdims dims(const int i0) {return Gdims(i0);}
  inline Gdims dims(const int i0, const int i1) {return Gdims(i0,i1);}
  inline Gdims dims(const int i0, const int i1, const int i2) {return Gdims(i0,i1,i2);}
  inline Gdims dims(const int i0, const int i1, const int i2, const int i3) {return Gdims(i0,i1,i2,i3);}
  inline Gdims dims(const int i0, const int i1, const int i2, const int i3, const int i4) {return Gdims(i0,i1,i2,i3,i4);}


  template<typename OBJ>
  class as_shape_tmp: public OBJ{
  public:
    as_shape_tmp(const OBJ& x, const Gdims& _dims): OBJ(x,fill::view){
      OBJ::reshape(_dims);}
  };


}


namespace std{

  template<>
  struct hash<cnine::Gdims>{
  public:
    size_t operator()(const cnine::Gdims& dims) const{
      size_t t=0;
      for(int i=0; i<dims.size(); i++) t=(t^hash<int>()(dims[i]))<<1;
      return t;
    }
  };

}



#endif
