#include "Cnine_base.cpp"
#include "ComplexHelper.hpp"

using namespace cnine;

template<typename TYPE>
class MyClass{
public:
  ComplexHelper<TYPE> is_conj;
  MyClass():is_conj(false){}
  bool conj(){return is_conj.flip();}
};

int main(int argc, char** argv){

  /*
  cout<<"not complex"<<endl;
  ComplexHelper<float> fhelper;
  cout<<fhelper()<<endl;
  fhelper.flip();
  cout<<fhelper()<<endl;

  cout<<"complex"<<endl;
  ComplexHelper<complex<double> > chelper;
  cout<<chelper()<<endl;
  chelper.flip();
  cout<<chelper()<<endl;
  */

  MyClass<float> a;
  a.conj();
  cout<<a.is_conj()<<endl;

  MyClass<complex<float> > b;
  b.conj();
  cout<<b.is_conj()<<endl;



}
