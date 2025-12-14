#include "Cnine_base.cpp"
#include "BGtensor.hpp"

using namespace cnine;


int main(int argc, char** argv){
  try{

    BGtensor<float> A(1,{},{2,2},4);
    cout<<A<<endl;

    cout<<BGtensor<float>::batched_scalar({1,2,3})<<endl;

    cout<<BGtensor<float>::tensor({{1,2,3},{4,5,6}})<<endl;

    cout<<BGtensor<float>::batched_tensor({{1,2,3},{4,5,6}})<<endl;

    cout<<BGtensor<float>::grid({{1,2,3},{4,5,6}})<<endl;



  }catch(const std::runtime_error& e){
    cerr<<"Error: "<<e.what()<<endl;
  }

}

