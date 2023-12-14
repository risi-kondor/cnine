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

#ifndef _GatherMapProgram
#define _GatherMapProgram

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "headed_lists.hpp"

namespace cnine{


  class GatherMapProgram;
  inline int makeGatherMapVar(GatherMapProgram&, const Gdims&);
  

  class GatherMapVar{
  public:

    friend class GatherMapProgram;

    const int id;

    int roffset=0;
    int coffset=0;

    GatherMapVar(GatherMapProgram& prog, const Gdims& _dims):
      id(makeGatherMapVar(prog,_dims)){}

    GatherMapVar operator()(const int _roffset, const int _coffset){
      return GatherMapVar(id,_roffset,_coffset);
    }

  private:

    GatherMapVar(const int _id, const int _roffset=0, const int _coffset=0):
      id(_id), roffset(_roffset), coffset(_coffset){};

  };



  
  class GatherMapProgram{
  public:


    class Variable{
    public:

      int id;
      const Gdims dims;


    public: // ---- Constructors --------------------------------------


      Variable(const int _id, const Gdims& _dims): 
	id(_id), dims(_dims){}


    public: // ---- I/O -----------------------------------------------


      string repr() const{
	ostringstream oss;
	if(id==0) oss<<"input";
	if(id==1) oss<<"output";
	if(id>1) oss<<"v"<<to_string(id);
	return oss.str();
      }

      string str() const{
	ostringstream oss;
	if(id==0) oss<<"input";
	if(id==1) oss<<"output";
	if(id>1) oss<<"v"<<id;
	oss<<dims;
	return oss.str();
      }

      friend ostream& operator<<(ostream& stream, const Variable& v){
	stream<<v.str(); return stream;}

    };
    

    class Instruction{
    public:

      int in;
      int in_roffset;
      int in_coffset;
      int out;
      int out_roffset;
      int out_coffset;
      shared_ptr<const GatherMapB> map;


    public: // ---- Constructors --------------------------------------


      //Instruction(const int _out, const int _in, shared_ptr<const GatherMapB> _map):
      //in(_in), out(_out), map(_map){}

      Instruction(const GatherMapVar& _out, const GatherMapVar& _in, shared_ptr<const GatherMapB> _map):
	in(_in.id), in_roffset(_in.roffset), in_coffset(_in.coffset), 
	out(_out.id), out_roffset(_out.roffset), out_coffset(_out.coffset), 
	map(_map){}


    public: // ---- I/O -----------------------------------------------


      string repr() const{
	return str();
      }

      string str() const{
	ostringstream oss;
	oss<<"V"<<to_string(out);
	if(out_roffset>0 || out_coffset>0) oss<<"["<<out_roffset<<","<<out_coffset<<"]";
	oss<<"<-"<<"gather(V"<<to_string(in);
	if(in_roffset>0 || in_coffset>0) oss<<"["<<in_roffset<<","<<in_coffset<<"]";
	oss<<")";
	return oss.str();
      }

      friend ostream& operator<<(ostream& stream, const Instruction& v){
	stream<<v.str(); return stream;}

    };


    vector<Variable> vars;
    vector<Instruction> instructions;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMapProgram(const Gdims& in_dims, const Gdims& out_dims){
      vars.push_back(Variable(0,in_dims));
      vars.push_back(Variable(1,out_dims));
    }


  public: // ---- Programming --------------------------------------------------------------------------------


    GatherMapVar input(){
      return GatherMapVar(vars[0].id);
    }

    GatherMapVar output(){
      return GatherMapVar(vars[1].id);
    }

    int add_var(const Gdims& _dims){
      vars.push_back(Variable(vars.size(),_dims));
      return vars.size()-1;
    }

    void gather(const GatherMapVar& out, const GatherMapVar& in, const GatherMapB* map){
      gather(out,in,shared_ptr<const GatherMapB>(map));
    }

    void gather(const GatherMapVar& out, const GatherMapVar& in, shared_ptr<const GatherMapB> map){
      instructions.push_back(Instruction(out,in,map));
    }
    

  public: // ---- Execution ----------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const TensorView<TYPE>& output, const TensorView<TYPE>& arg0){
    }


  public: // ---- I/O -----------------------------------------------
    

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Variables:"<<endl;
      for(auto& p:vars)
	oss<<indent<<"  "<<p<<endl;
      oss<<indent<<"Instructions:"<<endl;
      for(auto& p:instructions)
	oss<<indent<<"  "<<p<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMapProgram& v){
      stream<<v.str(); return stream;}
    

  };


  inline int makeGatherMapVar(GatherMapProgram& p, const Gdims& dims){
    return p.add_var(dims);
  }


}

#endif 
