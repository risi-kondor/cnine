#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Tensor.hpp"
#include "TensorArray.hpp"


using namespace cnine;

int main(int argc, char** argv){
  cnine_session session;
  cout<<endl;

  Gdims adims({2});
  Gdims ddims({3,3});

  TensorArray<float> u=TensorArray<float>::gaussian(adims,ddims);
  TensorArray<float> v=TensorArray<float>::gaussian(adims,ddims);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  cout<<u(0)<<endl;

  Tensor<float> a=Tensor<float>::sequential({2,3});
  cout<<TensorArray<float>({2,2},a)<<endl;

  cout<<endl; 
}
