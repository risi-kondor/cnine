/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


// ---- Products with scalars ------------------------------------------------------------------------------------------------


//template<typename TYPE2, typename = typename std::enable_if<std::is_same<TYPE2,float>::value, TYPE2>::type>
//template<typename TYPE2, std::enable_if_t<std::is_arithmetic_v<TYPE2>> = 0>
template<typename TYPE2, std::enable_if_t<is_numeric_or_complex_v<TYPE2>>* = nullptr>
BGtensor<TYPE> operator*(const TYPE2 c) const{
  return like(TensorView<TYPE>::operator*(c));
}


// ---- Elementwise products -------------------------------------------------------------------------------------------------


BGtensor<TYPE> operator*(const BGtensor<TYPE>& y) const{
  BGtensor<TYPE> R(dominant_batch(*this,y),dominant_gdims(*this,y),dominant_cdims(*this,y),0,get_dev());
  R.add_prod(*this,y);
  return R;
}


void add_prod(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  if(!x.has_cells()){
    if(!y.has_cells()){CNINE_UNIMPL(); return;}
    //ForEachCellMultiScalar<TYPE,TYPE,TYPE>()
    for_each_cell_multi_scalar(*this,y,x,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& y, const TYPE c){
	r.add(y,c);},0);
  }else if(!y.has_cells()){
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TYPE c){
	r.add(x,c);},0);
  }else{
    for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_prod(x,y);},0);
  }
}


void add_prod_XC(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  if(x.has_cells()+y.has_cells()+has_cells()<2){CNINE_UNIMPL(); return;}
  if(!x.has_cells()){
    for_each_cell_multi_scalar(*this,y,x,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& y, const TYPE c){
	r.add_CX(y,c);},0);
  }else if(!y.has_cells()){
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TYPE c){
	r.add(x,std::conj(c));},0);
  }else if(!has_cells()){
    for_each_cell_multi_scalar(x,y,*this,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& x, const TensorView<TYPE>& y, TYPE& r){
	r+=inp(y,x);
      },2);
  }else{
    for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_prod_XC(x,y);
      },0);
  }
}


// ---- Matrix products -----------------------------------------------------------------------------------------------------

  
BGtensor<TYPE> mprod(const BGtensor<TYPE>& y) const{
  if(ncdims()!=2||y.ncdims()!=2) CNINE_ARG_ERR("matrix product: both BGtensors must have cell dimensions 2");
  if(cdim(1)!=y.cdim(0)) CNINE_ARG_ERR("matrix product: inner dimensions do not match");
  BGtensor<TYPE> R(dominant_batch(*this,y),dominant_gdims(*this,y),Gdims({cdim(0),y.cdim(1)}),0,get_dev());
  R.add_mprod(*this,y);
  return R;
}

void add_mprod(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod(x,y);},0);
}

void add_mprod_AH(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod_AH(x,y);},0);
}

