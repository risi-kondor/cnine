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


#ifndef _CnineTensorSarrView
#define _CnineTensorSarrView

#include "Cnine_base.hpp"
#include <unordered_map>

#include "Hvector.hpp"
#include "Gdims.hpp"
#include "IntTensor.hpp"
#include "RtensorA.hpp"
#include "CSRmatrix.hpp"


namespace cnine{


  template<typename TYPE>
  class TensorSArrayView{
  public:

    typedef TensorView<TYPE> TensorView;

    MemArr<TYPE> arr;
    SparseTensor<int> offs;
    //Gdims adims;
    Gdims ddims;
    Gdims dstrides;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorSArrayView(const MemArr<TYPE>& _arr, const SparseTensor<int>& _offs, const Gdims& _ddims, const GstridesB& _dstrides):
      arr(_arr),
      offs(_offs),
      ddims(_ddims),
      dstrides(_dstrides){}


  public: // ---- Access -------------------------------------------------------------------------------------
    

    TensorView operator(const Gindex& ix) const{
      CNINE_ASSRT(offs.is_filled(ix));
      return TensorView(arr+offs(ix),ddims,dstrides);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_cell(const std::function<void(const Gindex&, const TensorView&)>& lambda) const{
      dir.for_each_nonzero([&](const Gindex& ix){
	  lambda(ix,(*this)(ix));
	});
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorSArrayView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each_cell([&](const Gindex& ix, const TensorView& x){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<x.str(indent+"  ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }




  }

}

#endif 
