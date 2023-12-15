#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapB.hpp"
#include "gather_rows.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  GatherMapB g=GatherMapB::random(10,10,0.2);
  cout<<g<<endl;
  cout<<g.sort()<<endl;
  cout<<g.inv()<<endl;

  Ltensor<int> A(dims(10,10));
  uniform_int_distribution<int> distr(0,4);
  for(int i=0; i<10; i++)
    for(int j=0; j<10; j++)
      A.set(i,j,distr(rndGen));
  cout<<A<<endl;

  Ltensor<int> B=GatherRows()(A,g);
  cout<<B<<endl;

  

}


