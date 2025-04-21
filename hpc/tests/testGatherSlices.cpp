#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapB.hpp"
#include "GatherRows.hpp"
#include "GatherSlices.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  GatherMapB g=GatherMapB::random(5,5,0.4);
  cout<<g<<endl;

  TensorView<float> X(dims(5,3),3,0);
  TensorView<float> Y(dims(5,3),0,0);

  GatherSlices()(Y,X,g,0);

  cout<<Y<<endl;

}
