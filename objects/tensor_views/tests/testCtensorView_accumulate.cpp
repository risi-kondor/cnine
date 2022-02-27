#include "Cnine_base.cpp"
#include "CtensorB.hpp"
#include "RtensorObj.hpp"
#include "Rmask1.hpp"
#include "AccumulateCmap.hpp"
#include "Ctensor1view_add.hpp"
#include "CnineSession.hpp"


using namespace cnine;

typedef RtensorObj rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){

  cnine_session genet;

  int n=5;
  device dev=deviceid::GPU0;
  cout<<endl;

  ctensor A=ctensor::zero({n,1});
  ctensor B=ctensor::sequential({n,1});

  rtensor M=rtensor::zero({n,n});
  M.set(0,1,1.0);
  M.set(0,3,1.0);
  M.set(2,2,1.0);
  cout<<M<<endl;

  Rmask1 mask=Rmask1::matrix(M.view2());
  cout<<mask<<endl;

  Ctensor1view_add op;
  AccumulateCmap(op,A.view2(),B.view2(),mask);
  print(A);

#ifdef _WITH_CUDA 

  //CtensorArray Ag=A.to(dev);
  //CtensorArray Bg=B.to(dev);

  //CtensorArray Cg(dims(4),dims(2,2),fill::zero,dev);
  //add_accumulate<CtensorA_plus>(mask,Cg,Ag,Bg);
  //printl("add_accumulate<CtensorA_plus>(mask,Cg,Ag,Bg)",Cg);

#endif 

}

