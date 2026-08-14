#include "Cnine_base.cpp"
#include "JSONlike.hpp"
#include "CnineSession.hpp"
#include "Gdims.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Gdims dims({4,3,5});
  JSONlike jlike0(dims);
  cout<<jlike0<<endl; 

  Gdims dims0({1,4});
  Gdims dims1({3,3});
  Gdims dims2({5,4});
  vector<vector<int> > Dims({dims0,dims1,dims2});

  JSONlike jlike1;
  jlike1["mode"]="forward";
  jlike1["mode2"]="insto";
  jlike1["dims"]=Dims;
  jlike1["strides"]=Dims;
  jlike1["vec"]={1,2,3};

  auto str1=jlike1.str("");
  cout<<str1<<endl<<endl;

  cout<<jlike1.str()<<endl<<endl;

  cout<<jlike1.lstr("")<<endl<<endl;;

  cout<<jlike1["vec"].get_vec<int>()<<endl<<endl;
  //auto z=jlike1["dims"].get_vec<string>();
  //for(auto& p:z) cout<<p<<endl;
  auto z=jlike1["dims"].get_vec_vec<int>();
  for(auto& p:z) cout<<p<<endl;
  cout<<endl;

  JSONlike jlike2(jlike1.str());
  cout<<jlike2.lstr("")<<endl;
  


}
