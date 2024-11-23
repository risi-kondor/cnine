#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "loop_tree_node.hpp"
#include "ctree.hpp"
#include "loop_tree.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ctree ctr;
  auto T0=ctr.add_input({0,1});
  auto T1=ctr.add_input({1,2});
  auto T2=ctr.add_input({2,3});
  auto T3=contract(T0,T1,1);
  auto T4=contract(T3,T2,2);

  loop_tree ltree(ctr);

  code_env env;
  ltree.write_to(env);
  cout<<env.str()<<endl;

}

