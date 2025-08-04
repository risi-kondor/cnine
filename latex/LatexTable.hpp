/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _LatexTable
#define _LatexTable

#include "TensorView.hpp"

namespace cnine{

  template<typename TYPE>
  class LatexTable{
  public:


    vector<string> row_headings;
    vector<string> col_headings;

    TensorView<TYPE> data;

    template<typename TYPE1>
    LatexTable(const vector<TYPE1>& rows, const vector<TYPE1>& cols, const TensorView<TYPE>& _data):
      data(_data){
      for(auto& p: rows)
	row_headings.push_back(to_string(p));
      for(auto& p: cols)
	col_headings.push_back(to_string(p));
    }

    string latex() const{
      ostringstream oss;
      oss<<"\\begin{tabular}{|l|";
      for(int j=0; j<data.dim(1); j++)oss<<"c|";
      oss<<"}"<<endl;
      oss<<"\\hline"<<endl;
      for(int j=0; j<data.dim(1); j++)
	oss<<"&"<<col_headings[j];
      oss<<"\\\\"<<endl;
      oss<<"\\hline"<<endl;
      for(int i=0; i<data.dim(0); i++){
	oss<<row_headings[i];
	for(int j=0; j<data.dim(1); j++)
	  oss<<"&"<<to_string(data(i,j));
	oss<<"\\\\"<<endl;
      }
      oss<<"\\hline"<<endl;
      oss<<"\\end{tabular}"<<endl;
      return oss.str();
    }

  };

    

}

#endif 
