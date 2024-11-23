#ifndef _loop_tree
#define _loop_tree

#include "ctree.hpp"
#include "loop_tree_node.hpp"

namespace cnine{


  class loop_tree{
  public:

    typedef loop_tree_index_set IXSET;

    shared_ptr<loop_tree_node> root;
    shared_ptr<loop_tree_tensor_registry> registry;

    loop_tree(){
      registry=to_share(new loop_tree_tensor_registry());
      root=to_share(new loop_tree_node(registry));
    }

    loop_tree(const ctree& _ctr):
      loop_tree(){
      for(int i=_ctr.nodes.size()-1; i>=0; i--)
	insert(_ctr.nodes[i]);
    }


  public: // ------------------------------------------------------------------------------------------------


    void insert(const ctree_tensor_handle& x){
      insert(x.obj);
    }

    void insert(const shared_ptr<ctree_tensor_node>& x){
      if(dynamic_pointer_cast<ctree_contraction_node>(x))
	insert_contraction(*dynamic_pointer_cast<ctree_contraction_node>(x));
      else
	root->insert_tensor(x->id,x->indices);
    }

    void insert_contraction(const ctree_contraction_node& x){
      vector<int> args;
      for(auto& p: x.args)
	args.push_back(p->id);
      loop_tree_contr lcontr(x.id,x.ix,args,x.dependents);
      root->insert(x.indices,lcontr);
    }


    void insert(vector<int> indices, const shared_ptr<ctree_tensor_node>& _x){
      if(dynamic_pointer_cast<ctree_contraction_node>(_x)){
	auto& x=*dynamic_pointer_cast<ctree_contraction_node>(_x);
	vector<int> args;
	for(auto& p: x.args)
	  args.push_back(p->id);
	loop_tree_contr lcontr(x.id,x.ix,args,x.dependents);
	root->insert(indices,lcontr);
      }
      else
	root->insert_tensor(_x->id,_x->indices);
    }



  public: // ------------------------------------------------------------------------------------------------


    void write_to(code_env& env){
      if(root)
	root->write_to(env);
    }

  };

}

#endif 
