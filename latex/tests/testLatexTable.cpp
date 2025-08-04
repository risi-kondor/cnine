#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LatexTable.hpp"
#include "LatexDoc.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;
  LatexDoc doc;

  vector<int> labels({0,1,2,3});
  TensorView<int> M({4,4},3,0);

  LatexTable<int> table(labels,labels,M);
  doc<<table.latex();

  doc.compile("table");

}
