#ifndef _ContractionNode
#define _ContractionNode

#include "EinsumForm.hpp"


namespace cnine{

  class ContractionNode{
  public:

    //shared_ptr<EinsumForm> form;
    
    int contraction_index;
    int arg_id=-1;
    token_string indices;
    index_set external_indices;
    index_set internal_indices;

    vector<shared_ptr<ContractionNode> > children;

    int level=0;
    string name="M";

    ContractionNode(){}
    
    ContractionNode(const int _arg_id, const token_string& _indices, const string _name="M"):
      arg_id(_arg_id),
      indices(_indices),
      external_indices(_indices),
      name(_name){}

    ContractionNode(int _contraction_index, vector<shared_ptr<ContractionNode> >& _children):
      contraction_index(_contraction_index){
      for(auto& p:_children){
	children.push_back(p);
	for(auto q:p->indices){
	  if(q!=contraction_index && !indices.contains(q)){
	    indices.push_back(q);
	  }
	}
	bump(level,p->level);
      }
      level++;
      for(auto p:indices){
	if([&](){for(auto& q:_children)
	       if(!q->indices.contains(p)) return false;
	     return true;}()) external_indices.insert(p); 
	else internal_indices.insert(p);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool contains(const int i) const{
      return indices.contains(i);
    }

    int n_ops(const vector<int>& dims) const{
      int t=0;
      //for(auto& p: children)
      //t+=p->n_ops(dims); // TODO 
      //return t+dims[id]*adims()*children.size();
      return 0;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string to_string(const int ix) const{
      return string(1,static_cast<char>('a'+ix));
    }

    string index_string() const{
      ostringstream oss;
      for(auto& p:indices)
	oss<<static_cast<char>('a'+p);
      //oss<<form->tokens(p);
      return oss.str();
    }

    void latex(ostream& oss) const{
      oss<<"\\Big(\\sum_"<<to_string(contraction_index)<<" ";
      for(auto p: children)
	p->latex(oss);
      oss<<"\\Big)";
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      if(children.size()>0){
	oss<<indent<<"contract "<<to_string(contraction_index)<<": ["<<
	  external_indices<<"/"<<internal_indices<<"]"<<endl;
	for(auto& p: children)
	  oss<<p->str(indent+"  ");
      }else{
	oss<<indent<<index_string()<<" ["<<external_indices<<"]"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const ContractionNode& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
