/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


// ---- Permutations -----------------------------------------------------------------------------------------


TensorView transp() const{
  CNINE_ASSRT(ndims()==2);
  return TensorView(arr,{dims[1],dims[0]},{strides[1],strides[0]});
}

TensorView transp(const int a, const int b) const{
  CNINE_ASSRT(a<ndims());
  CNINE_ASSRT(b<ndims());
  return TensorView(arr,dims.copy().set(a,dims[b]).set(b,dims[a]),strides.copy().set(a,strides[b]).set(b,strides[a]));
}

TensorView permute_indices(const vector<int>& p){
  return TensorView(arr,dims.permute(p),strides.permute(p));
}

TensorView falling_strides(){
  auto p=strides.descending_ordering();
  return TensorView(arr,dims.permute(p),strides.permute(p));
}

TensorView with_decreasing_strides(){
  auto p=strides.descending_ordering();
  return TensorView(arr,dims.permute(p),strides.permute(p));
}


// ---- Reshaping -------------------------------------------------------------------------------------------


TensorView reshape(const Gdims& _dims) const{
  // add option for non-regular tensors
  CNINE_ASSRT(_dims.asize()==asize());
  CNINE_ASSRT(is_regular());
  return TensorView<TYPE>(arr,_dims,GstridesB(_dims));
}

TensorView inplace_reshape(const Gdims& _dims){
  CNINE_ASSRT(_dims.asize()==asize());
  CNINE_ASSRT(is_regular());
  dims=_dims;
  strides=GstridesB(_dims);
  return *this;
}

TensorView unsqueeze(const int d) const{
  int s=ifthen(d==0,strides.memsize(dims),strides[d-1]);
  return TensorView(arr,Gdims(dims).insert(d,1),GstridesB(strides).insert(d,s));
}

TensorView  insert_dim(const int d, const int n) const{
  return TensorView(arr,Gdims(dims).insert(d,n),GstridesB(strides).insert(d,0));
}

TensorView scrunch() const{
  int n=ndims();
  Gdims _dims;
  GstridesB _strides;
  auto ordering=strides.ordering();

  for(int i=0; i<n; i++){
    int s=strides[ordering[i]];
    int d=dims[ordering[i]];
    while(i+1<n && strides[ordering[i+1]]==s*d){
      d*=dims[ordering[i+1]];
      i++;
    }
    _dims.push_back(d);
    _strides.push_back(s);
  }

  std::reverse(_dims.begin(),_dims.end());
  std::reverse(_strides.begin(),_strides.end());

  return TensorView(get_arr(),_dims,_strides);
}


pair<TensorView,TensorView> co_scrunch(const TensorView& y) const{
  CNINE_ASSRT(dims==y.dims);
  int n=ndims();
  Gdims _dims;
  GstridesB _xstrides;
  GstridesB _ystrides;

  auto ordering=strides.ordering();

  for(int i=0; i<n; i++){
    int xs=strides[ordering[i]];
    int ys=y.strides[ordering[i]];
    int d=dims[ordering[i]];
    while(i+1<n && strides[ordering[i+1]]==xs*d && y.strides[ordering[i+1]]==ys*d){
      d*=dims[ordering[i+1]];
      i++;
    }
    _dims.push_back(d);
    _xstrides.push_back(xs);
    _ystrides.push_back(ys);
  }

  std::reverse(_dims.begin(),_dims.end());
  std::reverse(_xstrides.begin(),_xstrides.end());
  std::reverse(_ystrides.begin(),_ystrides.end());

  return make_pair(TensorView(get_arr(),_dims,_xstrides),TensorView(y.get_arr(),_dims,_ystrides));
}


// ---- Fusing and splitting  -------------------------------------------------------------------------------


TensorView fuse01() const{
  CNINE_ASSRT(ndims()>=2);
  Gdims d=dims.remove(1); 
  d[0]*=dims[1];
  GstridesB s=strides.remove(0);
  return TensorView(arr,d,s);
}

TensorView fuse(const int a, const int b) const{
  CNINE_ASSRT(a<ndims());
  CNINE_ASSRT(b<ndims());
  if(strides(a)<strides(b)){
    CNINE_ASSRT(dims[a]*strides[a]==strides[b]);
    return TensorView(arr,dims.copy().set(a,dims(a)*dims(b)).remove(b),strides.remove(b));
  }else{
    CNINE_ASSRT(dims[b]*strides[b]==strides[a]);
    return TensorView(arr,dims.copy().set(b,dims(a)*dims(b)).remove(a),strides.remove(a));
  }
}

TensorView fuse(const vector<int>& ix) const{ // TODO 
  CNINE_ASSRT(ix.size()>0);
  CNINE_ASSRT(ix[0]<dims.size());
  int t=dims[ix[0]];
  int s=strides[ix[0]];
  for(int i=1; i<ix.size(); i++){
    CNINE_ASSRT(ix[i]<dims.size());
    t*=dims[ix[i]];
    if(strides[ix[i]]<s) s=strides[ix[i]];
  }
  vector<int> ix0(ix.begin()+1,ix.end());
  return TensorView(arr,dims.remove(ix0).set(ix[0],t),strides.remove(ix0));
}

TensorView split0(const int a, const int b) const{
  CNINE_ASSRT(ndims()>=1);
  CNINE_ASSRT(dims[0]==a*b);
  Gdims d=dims.insert(0,a); 
  d[1]=b; 
  GstridesB s=strides.insert(0,strides[0]*b);
  return TensorView(arr,d,s);
}

TensorView split1(const int a, const int b) const{
  CNINE_ASSRT(ndims()>=2);
  CNINE_ASSRT(dims[1]==a*b);
  Gdims d=dims.insert(1,a); 
  d[2]=b; 
  GstridesB s=strides.insert(1,strides[1]*b);
  return TensorView(arr,d,s);
}

TensorView split(const int d, const int a) const{
  CNINE_ASSRT(ndims()>d);
  CNINE_ASSRT(dims[d]%a==0);
  Gdims D=dims.insert(d+1,a); 
  D[d]/=a;
  GstridesB S=strides.insert(d,strides[d]*a);
  return Ltensor(arr,D,S);
}

TensorView<TYPE> cinflate(const int d, const int n) const{
  CNINE_ASSRT(dims[d]==1||dims[d]==n);
  if(dims[d]==n) return *this; 
  TensorView<TYPE> R(*this);
      R.dims[d]=n;
      R.strides[d]=0;
      return R;
}


// ---- Tensor products -------------------------------------------------------------------------------------


TensorView<TYPE> tprod_view(const Gdims& dims1, const Gdims& dims2) const{
  int k=ndims();
  CNINE_ASSRT(dims1.size()==k);
  CNINE_ASSRT(dims2.size()==k);
  Gdims rdims(2*k,fill_raw());
  for(int i=0; i<k; i++){
    rdims[2*i]=dims1[i];
    rdims[2*i+1]=dims2[i];
  }
  GstridesB rstrides(2*k,fill_raw());
  for(int i=0; i<k; i++){
    rdims[2*i]=dims2[i]*strides[i];
    rdims[2*i+1]=strides[i];
  }
  return TensorView<TYPE>(arr,rdims,rstrides);
}

