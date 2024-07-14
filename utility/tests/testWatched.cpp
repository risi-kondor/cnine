#include "Cnine_base.cpp"
#include "watched.hpp"
#include "CnineSession.hpp"


using namespace cnine;


class Widget;

watcher<int> int_watcher;


class Widget{
public:

  int id;

  watched<int> member=watched<int>(int_watcher,[&](){
      //int* a=new int(2*id); 
      return make_shared<int>(2*id);});

  Widget(const int x):
    id(x){};

};


int main(int argc, char** argv){

  cnine_session session;

  Widget w1(1);
  Widget w2(2);
  Widget* w3=new Widget(3);
  cout<<int_watcher<<endl;

  int b=w1.member;
  cout<<b<<endl;
  cout<<int_watcher<<endl;

  int c=w2.member;
  cout<<c<<endl;
  cout<<int_watcher<<endl;

  cout<<w3->member<<endl;
  cout<<int_watcher<<endl;

  delete w3;
  cout<<int_watcher<<endl;
}

