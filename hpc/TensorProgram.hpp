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

#ifndef _TensorProgram
#define _TensorProgram

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "GatherMapProgramHelpers.hpp"
#include "GatherRows.hpp"


namespace cnine{


  class TensorProgramVar{
  public:

    Gdims dims;

    TensorProgVar(const Gdims& _dims):
      dims(_dims){
      CNINE_ASSRT(dims.size()=2);
    }

  };


  template<typename OP>
  class TensorProgInstr{
  public:

    int in;
    int out;
    shared_ptr<OP> map;

    TensorProgInstr(const shared_ptr<OP>& map, const int _out, const int _in):
      in(_in), out(_out), map(_map);

  };




  template<typename OP>
  class TensorProgram{
  public: 

    typedef TensorProgramVariable VAR;
    typedef GatherMapProgramInstruction INST;


    vector<Variable> vars;
    vector<Instruction> instructions;
    int is_inverse=false;




    TensorMapProgram<OP> inv() const{
      TensorMapProgram<OP> R;

      R.vars.resize(vars.size());
      R.vars[0]=vars[1];
      R.vars[1]=vars[0];
      for(int i=2; i<vars.size(); i++)
	R.vars[i]=vars[i];

      int ninst=instructions.size();
      R.instructions.resize(ninst);
      for(int i=0; i<ninst; i++){
	R.instructions[i]=instructions[ninst-1-i].inv();
      }

      return R;
    }


  public: // ---- Execution ----------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const Tensor<TYPE>& output, const Tensor<TYPE>& arg0){
      CNINE_ASSRT(output.get_dev()==arg0.get_dev());
      CNINE_ASSRT(arg0.ndims()==2);
      CNINE_ASSRT(output.ndims()==2);
      int nc=arg0.dim(2);
      int dev=output.get_dev();

      vector<Ltensor<TYPE>*> v(vars.size());
      v[0]=new Ltensor<TYPE>(arg0);
      v[1]=new Ltensor<TYPE>(output);

      for(int i=2; i<vars.size(); i++)
	v[i]=new Ltensor<TYPE>(Gdims(vars[i].dims[0],vars[i].dims[1],ncols),0,dev);

      for(auto& p:instructions){
	CNINE_ASSRT(p.out<vars.size());
	CNINE_ASSRT(p.in<vars.size());
	GatherRows()(*v[p.out],*v[p.in],*p.map);
      }

      for(auto p:v) 
	delete p;
    }

  };

}

#endif 
