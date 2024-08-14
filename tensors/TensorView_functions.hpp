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


#ifndef _Cnine_TensorView_functions
#define _Cnine_TensorView_functions

namespace cnine{


  template<typename TYPE>
  TYPE inp(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return x.inp(y);
  }

  template<typename TYPE>
  TYPE norm2(const TensorView<TYPE>& x){
    return x.norm2();
  }

  template<typename TYPE>
  TYPE norm(const TensorView<TYPE>& x){
    return x.norm();
  }


  // ---- View converters ----------------------------------------------------


  inline Itensor1_view view1_of(const TensorView<int>& x){
    return Itensor1_view(x.mem(),x.dim(0),x.stride(0),x.get_dev());}
  inline Rtensor1_view view1_of(const TensorView<float>& x){
    return Rtensor1_view(x.mem(),x.dim(0),x.stride(0),x.get_dev());}
  inline Rtensor1_view view1_of(const TensorView<double>& x){// hack!!
    return Rtensor1_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.stride(0),x.get_dev());}
  inline Ctensor1_view view1_of(const TensorView<complex<float> >& x){
    return Ctensor1_view(x.mem_as<float>(),x.mem_as<float>()+1,x.dim(0),2*x.stride(0),x.get_dev());}

  inline Itensor2_view view2_of(const TensorView<int>& x){
    return Itensor2_view(x.mem(),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Rtensor2_view view2_of(const TensorView<float>& x){
    return Rtensor2_view(x.mem(),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Rtensor2_view view2_of(const TensorView<double>& x){ // hack!!
    return Rtensor2_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Ctensor2_view view2_of(const TensorView<complex<float> >& x){
    return Ctensor2_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),2*x.stride(0),2*x.stride(1),x.get_dev());}

  inline Itensor3_view view3_of(const TensorView<int>& x){
    return Itensor3_view(x.mem(),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Rtensor3_view view3_of(const TensorView<float>& x){
    return Rtensor3_view(x.mem(),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Rtensor3_view view3_of(const TensorView<double>& x){
    return Rtensor3_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Ctensor3_view view3_of(const TensorView<complex<float> >& x){
    return Ctensor3_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),2*x.stride(0),2*x.stride(1),2*x.stride(2),x.get_dev());}

  inline Rtensor1_view flat_view_of(const TensorView<float>& x){
    CNINE_ASSRT(x.is_contiguous());
    return Rtensor1_view(x.mem(),x.asize(),1,x.get_dev());}

}


#endif 
