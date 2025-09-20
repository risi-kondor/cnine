#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Convolve2D.hpp"
#include "TensorView_functions.hpp"

using namespace cnine;

typedef TensorView<float> RTENSOR;

void test_backprop0(const RTENSOR& x, const RTENSOR& w){

  auto r=convolve2d(x,w);
  auto xeps=x.gaussian_like();
  auto testg=r.gaussian_like();

  auto xg=x.zeros_like();
  Convolve2D_back0()(testg,xg,w);
  auto rd=convolve2d(x+xeps,w);

  cout<<xeps.inp(xg)<<endl;
  cout<<rd.inp(testg)-r.inp(testg)<<endl;
}


void test_backprop1(const RTENSOR& x, const RTENSOR& w){

  auto r=convolve2d(x,w);
  auto weps=w.gaussian_like();
  auto testg=r.gaussian_like();

  auto wg=w.zeros_like();
  Convolve2D_back1()(testg,x,wg);
  auto rd=convolve2d(x,w+weps);

  cout<<weps.inp(wg)<<endl;
  cout<<rd.inp(testg)-r.inp(testg)<<endl;
}


int main(int argc, char** argv){

  cnine_session session;

  TensorView<float> x({10,10},2,0);
  TensorView<float> w({4,4},2,0);

  auto r=convolve2d(x,w,0);
  cout<<r<<endl;

  if(true){
    TensorView<float> x({10,10},4,0);
    TensorView<float> w({4,4},4,0);
    test_backprop0(x,w);
    test_backprop1(x,w);
    cout<<endl;
  }

  if(true){
    TensorView<float> x({10,10,2},4,0);
    TensorView<float> w({4,4,3,2},4,0);
    test_backprop0(x,w);
    test_backprop1(x,w);
    cout<<endl;
  }

  if(true){
    TensorView<float> x({10,10,2,3},4,0);
    TensorView<float> w({4,4,3,2},4,0);
    test_backprop0(x,w);
    test_backprop1(x,w);
    cout<<endl;
  }

}
