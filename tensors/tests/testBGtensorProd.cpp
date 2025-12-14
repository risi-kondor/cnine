#include "Cnine_base.cpp"
#include "BGtensor_functions.hpp"

using namespace cnine;


int main(int argc, char** argv){
  try{

    auto bvec=BGtensor<float>::batched_scalar({1,2,3});
    cout<<bvec<<endl;

    auto grid=BGtensor<float>::grid({{1,2},{3,4}});
    cout<<grid<<endl;

    auto cell=BGtensor<float>::tensor({{1,2},{3,4}});
    cout<<cell<<endl;

    auto z0=bvec*cell;
    cout<<"z0=\n"<<z0<<endl;

    auto z1=grid*cell;
    cout<<"z1=\n"<<z1<<endl;

    auto z2=cell*cell;
    cout<<"z2=\n"<<z2<<endl;

    auto z3=bvec*z1;
    cout<<"z3=\n"<<z3<<endl;

    
  }catch(const std::runtime_error& e){
    cerr<<"Error: "<<e.what()<<endl;
  }

}

