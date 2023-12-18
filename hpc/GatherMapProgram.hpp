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
#include "GatherMapProgramHelpers.hpp"


namespace cnine{


  class GatherMapProgram{
  public: 

    typedef GatherMapProgramVariable Variable;
    typedef GatherMapProgramInstruction Instruction;


    vector<Variable> vars;
    vector<Instruction> instructions;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMapProgram(const Gdims& in_dims, const Gdims& out_dims){
      vars.push_back(Variable(0,in_dims));
      vars.push_back(Variable(1,out_dims));
    }

    GatherMapProgram(const Gdims& in_dims, const Gdims& out_dims, const GatherMapB& g){
      vars.push_back(Variable(0,in_dims));
      vars.push_back(Variable(1,out_dims));
      instructions.push_back(Instruction(1,0,g));
    }


    GatherMapProgram(const GatherMapB* g){
      vars.push_back(Variable(0));
      vars.push_back(Variable(1));
      instructions.push_back(Instruction(g,1,0));
    }

    GatherMapProgram(const GatherMapB& g){
      vars.push_back(Variable(0));
      vars.push_back(Variable(1));
      instructions.push_back(Instruction(g,1,0));
    }



  public: // ---- Programming --------------------------------------------------------------------------------


    GatherMapVar input(const int i=0){
      return GatherMapVar(vars[0].id);
    }

    GatherMapVar output(const int i=0){
      return GatherMapVar(vars[1].id);
    }

    int add_var(const Gdims& _dims){
      vars.push_back(Variable(vars.size(),_dims));
      return vars.size()-1;
    }


    int add_map(const GatherMapB* map, const int out=1, const int in=0){
      instructions.push_back(Instruction(map,out,in));
    }

    int add_map(const GatherMapB& map, const int out=1, const int in=0){
      instructions.push_back(Instruction(map,out,in));
    }


    [[deprecated]]
    void gather(const GatherMapVar& out, const GatherMapVar& in, const GatherMapB* map){
      gather(out,in,shared_ptr<const GatherMapB>(map));
    }

    [[deprecated]]
    void gather(const GatherMapVar& out, const GatherMapVar& in, shared_ptr<const GatherMapB> map){
      instructions.push_back(Instruction(map,out,in));
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
