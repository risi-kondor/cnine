#include "Cnine_base.cpp"
#include "TensorView.hpp"
#include "TensorView_functions.hpp"
#include "CnineSession.hpp"
#include "TensorAccessor.hpp"

#include "TensorAccessorBLAS.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  auto A=TensorView<float>(cdims=Gdims({5,5}),filltype=3,device=0); 
  auto R=TensorView<float>(cdims=Gdims({5,5}),filltype=0,device=0); 

  cout<<A<<endl;

  TensorAccessor<float,2> Aa(A);
  TensorAccessor<float,2> Ra(R);

  add_gemm(Ra,Aa,Aa);

  cout<<R<<endl;
  
}


