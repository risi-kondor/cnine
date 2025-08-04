#ifndef _CnineTikzPlot
#define _CnineTikzPlot

#include "TikzStream.hpp"


namespace cnine{


  class TikzPlot{
  public:

    vector<int> x_vals;
    TensorView<int> data;

    TikzPlot(const vector<int>& _x, const TensorView<int> _data):
      x_vals(_x), data(_data){}


    string latex() const{
      TikzStream ots;
      ots<<"\\begin{tikzpicture}";
      ots<<"\\begin{axis}[";
      ots<<"xlabel={$x$}, ";
      ots<<"ylabel={$y$}, ";
      ots<<"grid=both ";
      ots<<"]\n";
      // ymin=0, area style, xlabel=x, ylabel=y]\n"; only marks
      ots<<"\\addplot[smooth, color=red, mark=*]\n";
      ots<<"coordinates {\n";
      for(int i=0; i<x_vals.size(); i++)
	ots<<"("<<x_vals[i]<<","<<data(0,i)<<")";
      ots<<"};\n";
      ots.write("\\end{axis}");
      ots.write("\\end{tikzpicture}");
      return ots.str();
    }

  };

}


#endif 
