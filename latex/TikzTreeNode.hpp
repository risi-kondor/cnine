#ifndef _CnineTikzTreeNode
#define _CnineTikzTreeNode

#include "TikzStream.hpp" 


namespace cnine{

  class TikzTreeNodeObj{
  public:

    string label;
    string rlabel;

    vector<shared_ptr<TikzTreeNodeObj> > children;

    TikzTreeNodeObj(const string _label, const string _rlabel=""):
      label(_label),
      rlabel(_rlabel){}

    shared_ptr<TikzTreeNodeObj> add_child(string _label="", const string _rlabel=""){
      auto r=make_shared<TikzTreeNodeObj>(_label,_rlabel);
      children.push_back(r);
      return r;
    }
     
    void write_latex(TikzStream& tstream){
      if(rlabel!="") tstream.write("node[label={[font=\\fontsize{6pt}{8pt}\\selectfont]right:"+rlabel+"}]{"+label+"}");
      else tstream.write("node{"+label+"}");
      //if(rlabel!="")
      //tstream.write("edge from parent node[right, text width=4cm, align=left] {"+rlabel+"}");
      for(auto& p:children){
	tstream.depth++;
	tstream.write("child{");
	p->write_latex(tstream);
	tstream.write("}");
	tstream.depth--;
      }
    }
 
  };


  class TikzTreeNode{
  public:

    shared_ptr<TikzTreeNodeObj> obj;

    TikzTreeNode(const shared_ptr<TikzTreeNodeObj> _obj):
      obj(_obj){}

    TikzTreeNode add_child(const string label="", const string rlabel=""){
      return obj->add_child(label,rlabel);
    }
    
  };

}

#endif 
