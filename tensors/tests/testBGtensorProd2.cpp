#include "Cnine_base.cpp"
#include "BGtensor_functions.hpp"
//#include "TensorView_functions.hpp"

using namespace cnine;


int main(int argc, char** argv){
  try{

    auto bvec=BGtensor<float>::batched_scalar({1,2,3});
    cout<<bvec<<endl;

    auto grid=BGtensor<float>::grid({{1,2},{3,4}});
    cout<<grid<<endl;

    auto cell=BGtensor<float>::tensor({{1,2},{3,4}});
    cout<<cell<<endl;

    auto z0=mprod(cell,cell);
    cout<<"z0=\n"<<z0<<endl;
    
  }catch(const std::runtime_error& e){
    cerr<<"Error: "<<e.what()<<endl;
  }

}
