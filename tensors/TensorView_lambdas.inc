/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


// ---- Lambdas -----------------------------------------------------------------------------------------------


void for_each(std::function<void(TYPE&)> lambda) const{
  CNINE_CPUONLY();
  TYPE* arr=get_arr();
  switch(ndims()){
  case 0:
    return;
  case 1:
    {
      int n0=dims[0];
      int s0=strides[0];
      for(int i0=0; i0<n0; i0++)
	return lambda(*(arr+i0*s0));
    }
    break;
  case 2:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  return lambda(*(arr+i0*s0+i1*s1));
    }
    break;
  case 3:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    return lambda(*(arr+i0*s0+i1*s1+i2*s2));
    }
    break;
  case 4:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      int n3=dims[3];
      int s3=strides[3];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      return lambda(*(arr+i0*s0+i1*s1+i2*s2+i3*s3));
    }
    break;
  case 5:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      int n3=dims[3];
      int s3=strides[3];
      int n4=dims[4];
      int s4=strides[4];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		return lambda(*(arr+i0*s0+i1*s1+i2*s2+i3*s3+i4*s4));
    }
    break;
  default:
    dims.for_each_index([&](const Gindex& ix){
	lambda(arr[strides.offs(ix)]);});
  }
}


void for_each(const std::function<void(const int i0, const int i1, TYPE& x)>& lambda) const{
  CNINE_ASSRT(ndims()==2);
  int n0=dims[0];
  int n1=dims[1];
  for(int i0=0; i0<n0; i0++)
    for(int i1=0; i1<n1; i1++)
      lambda(i0,i1,arr[strides.offs(i0,i1)]);
}

void for_each(const std::function<void(const Gindex&, TYPE& x)>& lambda) const{
  dims.for_each_index([&](const Gindex& ix){
      lambda(ix,const_cast<MemArr<TYPE>&>(arr)[strides.offs(ix)]);});
}


