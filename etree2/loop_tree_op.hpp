#ifndef _loop_tree_op
#define _loop_tree_op

#include "code_env.hpp"
#include "loop_tree_index_set.hpp"


namespace cnine{


  class loop_tree_tensor_registry: public unordered_map<int,loop_tree_index_set >{
  public:
    
  };


  class loop_tree_contr{
  public:

    int id;
    int ix;
    //loop_tree_index_set indices;
    vector<int> args;
    vector<int> dependents;

    //const loop_tree_index_set& _indices):
    loop_tree_contr(const int _id, const int _ix, const vector<int>& _args, const vector<int>& _dependents): 
      id(_id),
      ix(_ix),
      args(_args),
      dependents(_dependents){}
      //indices(_indices){}
    
  };


  class loop_tree_tensor_node{
  public:

    int id;
    int ix;
    loop_tree_index_set indices;
    
    loop_tree_tensor_node(const int _id, const loop_tree_index_set& _indices):
      id(_id),
      indices(_indices){}


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){
      //string limits;
      //for(auto& p:indices)
      //limits+="n"+to_string(p)+",";

      env.write("TensorView<float> T"+to_string(id)+"("+indices.limit_str()+");");
    }

  };


  class loop_tree_op_node{
  public:

    shared_ptr<loop_tree_tensor_registry> registry;

    int id;
    int ix;
    //loop_tree_index_set indices;
    vector<int> args;
    //vector<int> dependents;
    
    
    loop_tree_op_node(const shared_ptr<loop_tree_tensor_registry> _registry,
      const int _id, const int _ix, const vector<int> _args):
      //const loop_tree_index_set& _indices):
      registry(_registry),
      id(_id),
      ix(_ix),
      args(_args){}
    //indices(_indices){}


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){
      string ixs=to_string(ix);
      env.add_line("float t=0;");
      env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
      env.depth++;
      string factrs;
      for(int i=0; i<args.size(); i++){
	factrs+="T"+to_string(args[i])+"("+(*registry)[args[i]].index_str()+")";
	if(i<args.size()-1) factrs+="*";
      }
	//auto& x=*factors[i];
      //factrs+="T"+to_string(x.tid)+"("+x.indices.str()+")";
      //if(i<factors.size()-1) factrs+="*";
      //}
      env.add_line("t+="+factrs+"");
      env.depth--;
      env.write("}");
      env.add_line("T"+to_string(id)+"("+(*registry)[id].index_str()+")+=t;");
    }

  };

}
#endif 
