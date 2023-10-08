/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _CnineFindPlantedSubgraphs
#define _CnineFindPlantedSubgraphs

#include "Cnine_base.hpp"
#include "sparse_graph.hpp"
#include "labeled_tree.hpp"
#include "labeled_forest.hpp"
#include "Tensor.hpp"
#include "flog.hpp"


namespace cnine{


  template<typename LABEL=int>
  class FindPlantedSubgraphs{
  public:

    typedef int_pool Graph;
    //typedef labeled_tree<int> labeled_tree;
    //typedef labeled_forest<int> labeled_forest;


    const Graph& G;
    const Graph& H;
    int n;
    int nmatches=0;
    vector<pair<int,int> > Htraversal;
    vector<int> assignment;
    Tensor<int> matches;

    int level;
    //vector<int> matching;
    //vector<int> pseudo_iterators;

  public:


    FindPlantedSubgraphs(const Graph& _G, const Graph& _H):
      G(_G), 
      H(_H), 
      n(_H.getn()),
      matches({1000,_H.getn()},fill_zero()){
      
      vector<int> Htraversal=H.depth_first_traversal();
      vector<int> parent_of(n);
      H.traverse([H&,parent_of&](const compact_int_tree::node& x){
	  parent_of[x.label()]=H.node_at(x.parent()).label();});
      vector<int> matching(n,-1);
      vector<int> psuedo_iterators(n,0);

      for(int i=0; i<n; i++){

	int level=0;
	int w=Htraversal[0];
	int v=i;

	while(level>=0){

	  int m1=H.size_of(w);
	  int m2=G.size_of(v);
	  bool success=true;

	  // check that every neighbor of w that is already part 
	  // of the matching corresponds to a neighbor of v 
	  for(int j=0; j<m1; j++){
	    int y=H(w,j);
	    if(matching[y]==-1) continue;
	    int vdash=matching[y];

	    bool found=false;
	    for(int p=0; p<m2; p++)
	      if(G(v,p)==vdash){
		found=true;
		break;
	      }
	    if(!found){
	      success=false;
	      break;
	    }
	  }

	  // check that every neighbor of v that is already part 
	  // of the matching corresponds to a neighbor of w 
	  if(success){
	    for(int j=0; j<m2; j++){
	      int vdash=H(v,j);
	      int wdash=-1;
	      for(int p=0; p<n; p++)
		if(matching[p]==vdash){
		  wdash=p;
		  break;
		}
	      if(wdash>=0){
		bool found=false;
		for(int p=0; p<m1; p++)
		  if(H(w,p)==wdash){
		    found=true;
		    break;
		  }
		if(!found){
		  success=false;
		  break;
		}
	      }
	    }
	  }

	  // if w has been successfully matched to v
	  if(success){
	    matching[w]=v;
	    matched[level]=v;
	    if(level==n-1){
	      add_match(matching);
	      success=false;
	    }
	  }

	  // if matched and not at the final level, try to descend
	  // even farther
	  if(success){
	    //auto hnode=Htree.node_at(Htraversal[level+1]);
	    //int neww=hnode.label();
	    //int parent_node=Htree.node_at(hnode.parent());
	    //int parentv=matching[parent_node.label()];
	    int parentv=matching[parent_of[Htraversal[level+1]]];
	    PTENS_ASSRT(parentv!=-1);
	    pseudo_iterators[level]=0;
	    int m3=G.size_of(parentv);
	    int newv=-1
	      for(int j=0; j<m3; j++){
		int candidate=G(parentv,j);
		for(int p=0; p<n; p++)
		  if(matching[p]==candidate){
		    found=true;
		    break;
		  }
		if(!found){
		  newv=candidate;
		  psuedo_iterators[level]=j+1;
		  break;
		}
	      }
	    if(newv>=0){
	      w=Htraversal[level+1];
	      v=newv;
	      level++;
	    }else{
	      success=false;
	    }
	  }

	  // if no match or could not descend farther forced to climb back
	  // and find alternative paths
	  if(!success){
	    matching[w]=-1;
	    level--;

	    while(level>=0){
	      int neww=Htraversal[level+1];
	      int parentv=matching[parent_of[neww]];
	      PTENS_ASSRT(parentv!=-1);
	      int m3=G.size_of(parentv);
	      int newv=-1;
	      for(int j=pseudo_iterators[level]; j<m3; j++){
		int candidate=G(parentv,j);
		for(int p=0; p<n; p++)
		  if(matching[p]==candidate){
		    found=true;
		    break;
		  }
		if(!found){
		  newv=candidate;
		  psuedo_iterators[level]=j+1;
		  break;
		}
	      }
	      if(newv!=-1){
		w=neww;
		v=newv;
		level++;
		break;
	      }
	      matching[Htraversal[level]]=-1
	      level--;
	    }
	  }

	}


      private:

	add_match(vector<int> matching){
	  CNINE_ASSRT(matching.size()==n);
	  std::sort(matching.begin(),matching.end());

	  for(int i=0; i<nmatches; i++){
	    bool is_same=true;
	    for(int j=0; j<n; j++)
	      if(matches(i,j)!=matching[j]){
		is_same=false; 
		break;
	      }
	    if(is_same) return;
	  }

	  CNINE_ASSRT(nmatches<matches.dim(0)-1);
	  for(int j=0; j<n; j++)
	    matches.set(nmatches,j,matching[j]);
	  nmatches++;
	}

      };

#endif 
