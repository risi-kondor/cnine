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

#ifndef _CnineTensorArrayPackView
#define _CnineTensorArrayPackView

#include "GElib_base.hpp"
#include "TensorPackView.hpp"
//#include "TensorArrayView.hpp"


namespace cnine{

  template<typename TYPE>
  class TensorArrayPackView: public cnine::TensorPackView<TYPE>{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorPackView<TYPE> TensorPackView;

    using TensorPackView::TensorPackView;
    using TensorPackView::dims;
    using TensorPackView::strides;
    using TensorPackView::arr;
    using TensorPackView::size;
    using TensorPackView::offset;


  public: // ---- Constructors --------------------------------------------------------------------------------

  public: // ---- Access --------------------------------------------------------------------------------------


    //TensorArrayView<RTYPE> operator[](const int i) const{
    //return TensorArrayView<RTYPE>(arr,dims(i),strides(i));
    //}


    //TensorArrayView<RTYPE> operator()(const int i){
    //return cnine::TensorView<complex<RTYPE> >(arr,dims(i),strides(i));
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayPackView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Array "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const TensorArrayPackView& v){
      stream<<v.str(); return stream;}


    

  };

}

#endif
