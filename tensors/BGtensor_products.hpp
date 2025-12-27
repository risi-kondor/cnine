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
  if(x.has_cells()+y.has_cells()+has_cells()<2){CNINE_UNIMPL(); return;}
  if(!x.has_cells()){
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

void add_prod_back0(const BGtensor<TYPE>& rg, const BGtensor<TYPE>& y) const{
  add_prod_XC(rg,y);
}

void add_prod_back1(const BGtensor<TYPE>& rg, const BGtensor<TYPE>& x) const{
  add_prod_XC(rg,x);
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
	r.add(x,sconj(c));},0);
  }else if(!has_cells()){
    for_each_cell_multi_scalar(x,y,*this,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& x, const TensorView<TYPE>& y, TYPE& r){
	r+=inp(y,x);},2);
  }else{
    for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_prod_XC(x,y);},0);
  }
}


// ---- Elementwise division ------------------------------------------------------------------------------------------------


BGtensor<TYPE> operator/(const BGtensor<TYPE>& y) const{
  BGtensor<TYPE> R(dominant_batch(*this,y),dominant_gdims(*this,y),dominant_cdims(*this,y),0,get_dev());
  R.add_div(*this,y);
  return R;
}

void add_div(const BGtensor& x, const BGtensor& y) const{
  try{
    if(y.has_cells()) CNINE_THROW("y cannot have cells.");
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TENSOR& r, const TENSOR& x, const TYPE c){
	r.add(x,((TYPE)(1.0))/c);},0);
  }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensor::add_div(x,y): ")+e.what())};
}

void add_div_back0(const BGtensor& x, const BGtensor& y) const{
  try{
    if(y.has_cells()) CNINE_THROW("y cannot have cells.");
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TENSOR& r, const TENSOR& x, const TYPE c){
	r.add(x,((TYPE)(1.0))/sconj(c));},0);
  }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensor::add_div_back0(x,y): ")+e.what())};
}

void add_div_back1_to(const BGtensor& yg, const BGtensor& x, const BGtensor& y) const{
  try{
    if(yg.has_cells()) CNINE_THROW("yg cannot have cells.");
    if(y.has_cells()) CNINE_THROW("y cannot have cells.");
    for_each_cell_multi_scalar2(*this,x,yg,y,
      [&](const int b, const Gindex& ix, const TENSOR& rg, const TENSOR& x, TYPE& yg, TYPE& y){
	yg-=inp(x,rg)/cnine::sconj(y*y);
      },2); 
  }catch(std::runtime_error& e) {CNINE_THROW(string("in BGtensor::add_div_back1_to(yg,x,y): ")+e.what())};
}


// ---- Matrix products -----------------------------------------------------------------------------------------------------

  
BGtensor<TYPE> mprod(const BGtensor<TYPE>& y) const{
  try{
    if(ncdims()!=2||y.ncdims()!=2) CNINE_THROW("both BGtensors must have cell dimensions 2");
    if(cdim(1)!=y.cdim(0)) CNINE_THROW(string("inner dimensions do not match between (")+repr()+", "+y.repr()+").");
    BGtensor<TYPE> R(dominant_batch(*this,y),dominant_gdims(*this,y),Gdims({cdim(0),y.cdim(1)}),0,get_dev());
    R.add_mprod(*this,y);
    return R;
  }catch(const std::runtime_error& e){CNINE_THROW(string("BGtensor::mprod(y): ")+e.what());}
}

void add_mprod(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod(x,y);},0);
}

void add_mprod_back0(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod_AH(x,y);},0);
}

void add_mprod_back1(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod_AH(x,y);},0);
}

void add_mprod_AH(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
      const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      r.add_mprod_AH(x,y);},0);
}



/*
void add_div(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  if(x.has_cells()+y.has_cells()+has_cells()<2){CNINE_UNIMPL(); return;}
  if(!x.has_cells()){
    //for_each_cell_multi_scalar(*this,y,x,[](const int b, const Gindex& ix,
    //const TensorView<TYPE>& r, const TensorView<TYPE>& y, const TYPE c){
    //r.add(y,c);},0);
  }else if(!y.has_cells()){
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TYPE c){
	r.add(x,1.0/c);},0);
  }else{
    for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_div(x,y);},0);
  }
}
*/

/*
void add_div_XC(const BGtensor<TYPE>& x, const BGtensor<TYPE>& y) const{
  if(x.has_cells()+y.has_cells()+has_cells()<2){CNINE_UNIMPL(); return;}
  if(!x.has_cells()){
    //for_each_cell_multi_scalar(*this,y,x,[](const int b, const Gindex& ix,
    //const TensorView<TYPE>& r, const TensorView<TYPE>& y, const TYPE c){
    //r.add_CX(y,c);},0);
  }else if(!y.has_cells()){
    for_each_cell_multi_scalar(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TYPE c){
	r.add(x,1.0/std::conj(c));},0);
  }else if(!has_cells()){
    //for_each_cell_multi_scalar(x,y,*this,[](const int b, const Gindex& ix,
    //const TensorView<TYPE>& x, const TensorView<TYPE>& y, TYPE& r){
    //r+=inp(y,x);},2);
  }else{
    for_each_cell_multi(*this,x,y,[](const int b, const Gindex& ix,
	const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	r.add_prod_XC(x,y);},0);
  }
}
*/
