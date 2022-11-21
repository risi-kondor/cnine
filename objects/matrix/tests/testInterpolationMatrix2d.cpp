#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "InterpolationMatrix2d.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  int n=5;

  RtensorA X=RtensorA::zero({5,2});
  InterpolationMatrix2d<float> M(X,6,6);

  cout<<M<<endl;
}
