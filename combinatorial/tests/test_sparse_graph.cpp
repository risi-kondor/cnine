
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "sparse_graph.hpp"
//#include "FindPlantedSubgraphs.hpp"

using namespace cnine;

typedef sparse_graph<int,float> Graph;


int main(int argc, char** argv){

  cnine_session session;

  Graph triangle(3,{{0,1},{1,2},{2,0}});
  Graph square(4,{{0,1},{1,2},{2,3},{3,0}});

  cout<<triangle.str()<<endl;

  Graph G=Graph::random(5,0.5);
  //Graph G(8);
  //G.insert(triangle,{0,1,2});
  //G.insert(triangle,{5,6,7});
  //cout<<G.dense();

  //auto fn=FindPlantedSubgraphs(G,triangle);
  //AindexPack sets(fn);
  //cout<<sets<<endl;

  //cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;
  //cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;

}

