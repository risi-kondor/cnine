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


#ifndef _CnineEinsumForm
#define _CnineEinsumForm

#include "TensorView.hpp"


namespace cnine{


  class token_string: public vector<int>{
  public:

    typedef vector<int> BASE;

    using BASE::BASE;

    bool contains(const int x) const{
      return std::find(begin(),end(),x)!=end();
    }

    int find(const int x) const{
      auto it=std::find(begin(),end(),x);
      CNINE_ASSRT(it!=end());
      return it-begin();
    }

  };

  class index_set: public set<int>{
  public:

    index_set(){}

    index_set(const vector<int>& x){
      for(auto p:x) 
	insert(p);
    }

    bool contains(const int x) const{
      return find(x)!=end();
    }

    vector<int> order(const vector<int>& loop_order){
      vector<int> r;
      for(auto& p:loop_order)
	if(contains(p)) r.push_back(p);
      return r;
    }

    string str() const{
      ostringstream oss;
      for(auto& p:*this)
	oss<<static_cast<char>('a'+p);
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const index_set& x){
      stream<<x.str(); return stream;
    }

  };


  class EinsumForm{
  public:

    map<string,int> dict; 
    vector<string> tokens;
    vector<token_string> args;
    vector<vector<int> > occurrences;

    vector<int> contraction_indices;

    vector<vector<int> > summations; // list of indices to sum for each arg
    vector<vector<pair<int,int> > > transfers;
    vector<vector<vector<int> > > map_to_dims;


    EinsumForm(const string str){

      auto d1=str.find("->");
      if(d1==string::npos)
	CNINE_ERROR(str+" is not a well formed einsum string because it has no rhs.");
      auto lhs=str.substr(0,d1);
      auto rhs=str.substr(d1+2,string::npos);

      int offs=0;
      vector<string> arg_str;
      while(offs<lhs.size()){
	auto d=lhs.find(",",offs);
	if(d==string::npos) d=lhs.size();
	arg_str.push_back(lhs.substr(offs,d-offs));
	offs=d+1;
      }

      auto [arg,mapping]=tokenize(rhs);
      args.push_back(arg);
      map_to_dims.push_back(mapping);

      for(auto& p:arg_str){
	auto [arg,mapping]=tokenize(p);
	args.push_back(arg);
	map_to_dims.push_back(mapping);
      }

      occurrences.resize(tokens.size());
      for(int i=0; i<tokens.size(); i++)
	for(int j=0; j<args.size(); j++)
	  if(std::find(args[j].begin(),args[j].end(),i)!=args[j].end())
	    occurrences[i].push_back(j);      

      int nargs=args.size();
      summations.resize(nargs);

      for(int i=0; i<tokens.size(); i++){
	auto& occ=occurrences[i];
	if(occ.size()==1){
	  summations[occ[0]].push_back(args[occ[0]].find(i));
	  continue;
	}
	if(occ[0]==0){
	  vector<pair<int,int> > v;
	  for(int j=0; j<occ.size(); j++)
	    v.push_back(pair<int,int>(occ[j],args[occ[j]].find(i)));
	  transfers.push_back(v);
	  continue;
	}
	if(true){
	  contraction_indices.push_back(i);
	}
      }

    }

  public: // ---- Access -------------------------------------------------------------------------------------


    string decode(const vector<int>& v) const{
      ostringstream oss;
      oss<<"(";
      for(auto p:v)
	oss<<tokens[p]<<",";
      if(v.size()>0) oss<<"\b";
      oss<<")";
      return oss.str();
    }


  private: // ------------------------------------------------------------------------------------------------


     pair<token_string,vector<vector<int> > > tokenize(const string& str){
      token_string tokenized;
      vector<vector<int> > mapping;

      map<string,vector<int> > r;
      int p=0;
      for(int i=0; p<str.size(); i++){
	if(str[p]!='('){
	  r[string({str[p]})].push_back(i);
	  p++;
	}else{
	  auto q=str.find_first_of(')',p+1);
	  if(q==string::npos) 
	    CNINE_ERROR("Unterminated '()'");
	  r[str.substr(p+1,q-p-1)].push_back(i);
	  p=q+1;
	}
      }

      for(auto& p:r){
	int id;
	if(dict.find(p.first)!=dict.end()){
	  id=dict[p.first];
	}else{
	  id=dict.size();
	  tokens.push_back(p.first);
	  dict[p.first]=id;
	}
	tokenized.push_back(id);
	mapping.push_back(p.second);
      }

      return make_pair(tokenized,mapping);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;

      for(int i=1; i<args.size(); i++){
	for(auto p:args[i])
	  if(tokens[p].size()==1) oss<<tokens[p];
	  else oss<<"("<<tokens[p]<<")";
	oss<<",";
      }
      oss<<"\b->";
      for(auto p:args[0])
	if(tokens[p].size()==1) oss<<tokens[p];
	else oss<<"("<<tokens[p]<<")";
      oss<<endl<<endl;

      for(int i=0; i<map_to_dims.size(); i++){
	oss<<"Mapping "<<i<<": ";
	for(auto& p:map_to_dims[i])
	  oss<<p<<",";
	oss<<"\b \n";
      }
      oss<<endl;

      oss<<"Summations: ";
      for(int i=1; i<args.size(); i++)
	oss<<decode(summations[i])<<",";
      oss<<"\b->";
      oss<<decode(summations[0]);
      oss<<endl;

      for(auto& p:transfers){
	oss<<"Transfer (";
	for(int j=1; j<p.size(); j++)
	  oss<<p[j].first<<":"<<tokens[p[j].second]<<",";
	oss<<"\b)->"<<tokens[p[0].second]<<endl;
      }
      oss<<endl;

      oss<<"Contraction indices: (";
	for(auto& p:contraction_indices)
	  oss<<tokens[p]<<",";
      oss<<"\b)\n";

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EinsumForm& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
