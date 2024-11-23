#ifndef _loop_tree_node
#define _loop_tree_node

#include "code_env.hpp"
#include "loop_tree_index_set.hpp"
#include "loop_tree_op.hpp"

namespace cnine{


  class loop_tree_node{
  public:

    int ix;
    shared_ptr<loop_tree_tensor_registry> registry;

    vector<shared_ptr<loop_tree_tensor_node> > tensors;
    vector<shared_ptr<loop_tree_op_node> > ops;
    vector<shared_ptr<loop_tree_node> > children;

    vector<int> indices;
    vector<int> ops_below;


    loop_tree_node(const shared_ptr<loop_tree_tensor_registry> _registry)://, const ctree* _ctr):
      registry(_registry),
      ix(-1){}

    loop_tree_node(const shared_ptr<loop_tree_tensor_registry> _registry, 
      //const ctree* _ctr, 
      const int _ix, const vector<int>& _indices):
      registry(_registry),
      ix(_ix),
      indices(_indices){
      indices.push_back(ix);
    }


  public: // ------------------------------------------------------------------------------------------------


    //void insert(const loop_tree_contr& x){
    //vector<int> r;
    //for(auto& p:x.args){
    //}
      //insert(vector<int>(x.indices),x);
    //}

    void insert_tensor(int _id, const loop_tree_index_set& _indices){
      auto new_tensor=new loop_tree_tensor_node(_id,_indices);
      //tensors.push_back(to_share(new_tensor));
      tensors.insert(tensors.begin(),to_share(new_tensor));
      (*registry)[_id]=_indices;
    }

    void insert(vector<int> remaining, const loop_tree_contr& x){
      ops_below.push_back(x.id);
      
      if(remaining.size()==0){
	auto r=new loop_tree_op_node(registry,x.id,x.ix,x.args);
	ops.push_back(to_share(r));
      }

      int xix=remaining[0];
      auto it=children.begin();
      while(it!=children.end()){
	auto& child=**it;
	if(child.ix==xix){
	  child.insert(vector<int>(remaining.begin()+1,remaining.end()),x);
	  return;
	}
	if([&](){
	    for(auto& p:x.dependents)
	      //if(child.ops_below.find(p)!=child.ops_below.end()) return true;
	      if(std::find(child.ops_below.begin(),child.ops_below.end(),p)!=child.ops_below.end()) return true;
	    return false;
	  }())
	  break;
	it++;
      }

      auto new_tensor=new loop_tree_tensor_node(x.id,remaining); //x.indices.minus(indices));
      tensors.push_back(to_share(new_tensor));
      (*registry)[x.id]=remaining;

      auto new_node=new loop_tree_node(registry,xix,indices);
      new_node->ops_below.push_back(x.id);
      children.insert(it,to_share(new_node));
      for(int j=1; j<remaining.size(); j++){
	auto nnew=new loop_tree_node(registry,remaining[j],new_node->indices);
	nnew->ops_below.push_back(x.id);
	new_node->children.push_back(to_share(nnew));
	new_node=nnew;
      }
      new_node->ops.push_back(to_share(new loop_tree_op_node(registry,x.id,x.ix,x.args)));
	//new_node->insert(vector<int>(remaining.begin()+1,remaining.end()),x);
    }


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){
      if(ix>=0){
	string ixs=to_string(ix);
	env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
	env.depth++;
      }

      for(auto& p: ops)
	p->write_to(env);
      
      for(auto& p: tensors)
	p->write_to(env);
      
      for(auto& p: children)
	p->write_to(env);

      if(ix>=0){
	env.depth--;
	env.write("}");
      }
    }

  };

}

#endif 
