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

#ifndef _BGtensor
#define _BGtensor

#include "TensorView.hpp"
#include "MultiLoop.hpp"

#include "ForEachCellMulti.hpp"
#include "ForEachCellMultiScalar.hpp"


namespace cnine{

  template<typename TYPE>
  class BGtensor: public TensorView<TYPE>{
  public:
  
    using TENSOR=TensorView<TYPE>;

    using TENSOR::arr;
    using TENSOR::dims;
    using TENSOR::strides;
    using TENSOR::ndims;
    using TENSOR::get_dev;
    using TENSOR::slice;
    using TENSOR::repr;

    //using TENSOR::TENSOR;

    int nc=0;

    /*
    BGtensor(const TENSOR& x):
      TENSOR(x){
      CNINE_ASSRT(dims.size()==nc || dims.size()==nc+1);
      if(dims.size()<nc+1){
	dims=dims.insert(0,1);
	strides=strides.insert(0,0);
      }
    }
    */

  public: // ---- Constructors -------------------------------------------------------------------------------


    BGtensor(){}

    BGtensor(const int b, const Gdims gdims, const Gdims cdims, const int fill=0, const int _dev=0):
      TENSOR(Gdims::cat(b,gdims,cdims),fill,_dev){
      nc=cdims.size();
    }

    BGtensor(const TENSOR& x, const bool _batched=0, const int _ngrid=0):
      TENSOR(x){
      nc=dims.size()-_batched-_ngrid;
      if(!_batched){
	dims=dims.insert(0,1);
	strides=strides.insert(0,0);
      }
    }

    BGtensor(const int _nc, const TENSOR& x):
      TENSOR(x){
      CNINE_ASSRT(_nc<=ndims()-1);
      nc=_nc;
    }

    BGtensor static like(const BGtensor& t, const TENSOR& x){
      return BGtensor(t.nc,x);
    }

    //static BGtensor (const TensorView<TYPE>& x):
    //BASE(x,true){}



    static BGtensor batched_scalar(const TENSOR& x){
      CNINE_ASSRT(x.ndims()==1);
      return BGtensor(x,1);
    }

    static BGtensor batched_scalar(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),true);
    }


    static BGtensor tensor(const TENSOR& x){
      return BGtensor(x,0);
    }

    static BGtensor tensor(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),0);
    }

    static BGtensor tensor(const initializer_list<initializer_list<TYPE> >& v){
      return BGtensor(TensorView<TYPE>::init(v),0);
    }


    static BGtensor batched_tensor(const TENSOR& x){
      return BGtensor(x,1);
    }

    static BGtensor batched_tensor(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),1);
    }

    static BGtensor batched_tensor(const initializer_list<initializer_list<TYPE> >& v){
      return BGtensor(TensorView<TYPE>::init(v),1);
    }


    static BGtensor grid(const TENSOR& x){
      return BGtensor(x,0,x.ndims());
    }

    static BGtensor grid(const initializer_list<TYPE>& v){
      return BGtensor(TensorView<TYPE>::init(v),0,1);
    }

    static BGtensor grid(const initializer_list<initializer_list<TYPE> >& v){
      return BGtensor(TensorView<TYPE>::init(v),0,2);
    }




  public: // ---- Conversions --------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims get_dims() const{
      return dims.chunk(-nc);
    }


  public: // ---- Batches ------------------------------------------------------------------------------------


    bool is_batched() const{
      return dims[0]>1;
    }

    int getb() const{
      return dims[0];
    }

    static int dominant_batch(const BGtensor& x, const BGtensor& y){
      int xb=x.getb();
      int yb=y.getb();
      if(xb==yb) return xb;
      if(xb==1) return yb;
      if(yb==1) return xb;
      throw std::invalid_argument("Cnine error: the batch dimensions of "+x.repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return 0;
    }

    static int dominant_batch(const BGtensor& x, const BGtensor& y, const BGtensor& z){
      int xb=x.getb();
      int yb=y.getb();
      int zb=z.getb();
      CNINE_ASSRT(xb>0 && yb>0 && zb>>0);

      int nb=((int)(xb>1))+((int)(yb>1))+((int)(zb>1));
      if(nb==0) return 1;
      if(nb==1) return (xb>1)*xb+(yb>1)*yb+(zb>1)*zb;
      if(nb==2){
	if(xb==1){
	  CNINE_ASSRT(yb==zb);
	  return yb;
	}
	if(yb==1){
	  CNINE_ASSRT(xb==zb);
	  return xb;
	}
	if(zb==1){
	  CNINE_ASSRT(xb==yb);
	  return xb;
	}
      }
      CNINE_ASSRT(xb==yb && xb==zb);
      return xb;
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_gridded() const{
      return dims.size()>nc+1;
    }

    bool is_grid() const{
      return dims.size()>nc+1;
    }

    bool has_grid() const{
      return dims.size()>nc+1;
    }

    int ngdims() const{
      return dims.size()-nc-1;
    }

    Gdims gdims() const{
      return dims.chunk(1,dims.size()-nc-1);
    }

    GstridesB gstrides() const{
      return strides.chunk(1,dims.size()-nc-1);
    }

    Gdims get_gdims() const{
      return dims.chunk(1,dims.size()-nc-1);
    }

    GstridesB get_gstrides() const{
      return strides.chunk(1,dims.size()-nc-1);
    }

    template<typename TYPE2>
    static Gdims dominant_gdims(const BGtensor& x, const BGtensor<TYPE2>& y){
      Gdims xg=x.gdims();
      Gdims yg=y.gdims();
      if(xg==yg) return xg;
      if(!x.is_grid()) return yg;
      if(!y.is_grid()) return xg;
      throw std::invalid_argument("Genet error: the grid dimensions of "+x.repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return Gdims();
    }

    static Gdims dominant_gdims(const BGtensor& x, const BGtensor& y, const BGtensor& z){
      Gdims xg=x.gdims();
      Gdims yg=y.gdims();
      Gdims zg=y.gdims();

      int ng=(int)(xg.size()>0)+(int)(yg.size()>0)+(int)(zg.size()>0);
      if(ng==0) return Gdims();
      if(ng==1){
	if(xg.size()>0) return xg;
	if(yg.size()>0) return yg;
	return zg;
      }
      if(ng==2){
	if(xg.size()==0){
	  CNINE_ASSRT(yg==zg);
	  return yg;
	}
	if(yg.size()==0){
	  CNINE_ASSRT(xg==zg);
	  return xg;
	}
	if(zg.size()==0){
	  CNINE_ASSRT(xg==yg);
	  return xg;
	}
      }
      CNINE_ASSRT(xg==yg && xg==zg);
      return xg;
    }


  public: // ---- Cells ---------------------------------------------------------------------------------------------------


    bool has_cells() const{
      return dims.size()>nc+1;
    }

    int ncdims() const{
      return nc;
    }

    Gdims cdims() const{
      if(nc==0) return Gdims();
      return dims.chunk(-nc);
    }

    GstridesB cstrides() const{
      if(nc==0) return GstridesB();
      return strides.chunk(-nc);
    }

    Gdims get_cdims() const{
      if(nc==0) return Gdims();
      return dims.chunk(-nc);
    }

    int cdim(const int i) const{
      CNINE_ASSRT(i<nc);
      return dims(-nc+i);
    }

    TENSOR cell(const int b, const Gindex& ix) const{
      return TENSOR(arr+b*strides[0]+get_gstrides().offs(ix),cdims(),cstrides());
    }

    template<typename TYPE2>
    static Gdims dominant_cdims(const BGtensor& x, const BGtensor<TYPE2>& y){
      Gdims xg=x.get_cdims();
      Gdims yg=y.get_cdims();
      if(xg==yg) return xg;
      if(x.nc==0) return yg;
      if(y.nc==0) return xg;
      throw std::invalid_argument("Genet error: the cell dimensions of "+x.repr()+" and "+y.repr()+" cannot be reconciled.");
      return Gdims();
    }


  public: // ---- Lambdas -------------------------------------------------------------------------------------------------


    template<typename TYPE2>
    void for_each_batch_multi(const BGtensor& x, const BGtensor<TYPE2>& y, 
			      const std::function<void(const int, const TENSOR& x, const TensorView<TYPE2>& y)>& lambda) const{
      if(x.getb()==1){
	int B=y.dim(0);
	for(int b=0; b<B; b++)
	  lambda(b,x.slice(0,0),y.slice(0,b));
	return;
      }
      if(y.dim(0)==1){
	MultiLoop(x.getb(),[&](const int b){
			   lambda(b,x.slice(0,b),y.slice(0,0));});
	return;
      }
      MultiLoop(x.getb(),[&](const int b){
			 lambda(b,x.slice(0,b),y.slice(0,b));});
    }


    void for_each_batch_multi(const BGtensor& x,  const BGtensor& y, const BGtensor& z, 
			      const std::function<void(const int, const TENSOR& x, const TENSOR& y, const TENSOR& z)>& lambda) const{
      int B=dominant_batch(x,y,z);
      if(x.getb()==1)
	for(int b=0; b<B; b++)
	  lambda(b,x.slice(0,0),y.slice(0,(y.dim(0)>1)*b),z.slice(0,(z.dim(0)>1)*b));
      else
	MultiLoop(B,[&](const int b){
			 lambda(b,x.slice(0,b),y.slice(0,(y.dim(0)>1)*b),z.slice(0,(z.dim(0)>1)*b));});
    }


    template<typename TYPE2>
    void for_each_batch_multi(const TensorView<TYPE2>& y, 
			      const std::function<void(const int, const TENSOR& x, const TensorView<TYPE2>& y)>& lambda) const{
      if(getb()==1){
	int B=y.dim(0);
	for(int b=0; b<B; b++)
	  lambda(b,slice(0,0),y.slice(0,b));
	return;
      }
      if(y.dim(0)==1){
	MultiLoop(getb(),[&](const int b){
			   lambda(b,slice(0,b),y.slice(0,0));});
	return;
      }
      MultiLoop(getb(),[&](const int b){
			 lambda(b,slice(0,b),y.slice(0,b));});
    }


  public: // ---- For each cell -----------------------------------------------------------------------------

    
    // this became a mess when allowing x and y to have different base types
    template<typename TYPE2>
    void for_each_cell_multi(const BGtensor<TYPE2>& y, 
			     const std::function<void(const int, const Gindex cell, const TENSOR& x, 
						      const TensorView<TYPE2>& y)> lambda) const{
      auto& x=*this;

      Gdims gdims=dominant_gdims(x,y);
      int ncells=gdims.asize();
      int kg=gdims.size();
      if(kg==0){ // shortcut 
	Gindex null_ix;
	for_each_batch_multi<TYPE2>(x,y,[&](const int b, const TENSOR& x, const TensorView<TYPE2>& y){
			       lambda(b,null_ix,x,y);});
	return;
      }

      TENSOR _x(x);
      GstridesB xstrides=GstridesB::zero(kg);
      if(x.strides.size()==nc+1+kg){
	xstrides=x.strides.chunk(0,kg);
	_x.dims=Gdims::cat(_x.dims[0],_x.dims.chunk(1+kg));
	_x.strides=Gdims::cat(_x.dims[0],_x.dims.chunk(1+kg));
      }

      TensorView<TYPE2> _y(y);
      GstridesB ystrides=GstridesB::zero(kg);
      if(x.strides.size()==nc+1+kg){
	xstrides=x.strides.chunk(0,kg);
  	_y.dims=Gdims::cat(_y.dims[0],_y.dims.chunk(1+kg));
	_y.strides=Gdims::cat(_y.dims[0],_y.dims.chunk(1+kg));
      }

      for_each_batch_multi<TYPE2>(_x,_y,[&](const int b, const TENSOR& x, const TensorView<TYPE2>& y){
			     TENSOR _x(x);
			     TensorView<TYPE2> _y(y);
			     for(int i=0; i<ncells; i++){
			       Gindex ix(i,gdims);
			       _x.arr=x.arr+xstrides.offs(ix);
			       _y.arr=y.arr+ystrides.offs(ix);
			       lambda(b,ix,_x,_y);
			     }
			   });
    }
      

    void for_each_cell_multi(const BGtensor& y, const BGtensor& z, 
			     const std::function<void(const int, const Gindex cell, const TENSOR& x, const TENSOR& y, const TENSOR& z)>& lambda) const{
      auto& x=*this;

      Gdims gdims=dominant_gdims(x,y,z);
      int ncells=gdims.asize();
      int kg=gdims.size();
      if(kg==0){ // shortcut 
	Gindex null_ix;
	for_each_batch_multi(x,y,z,[&](const int b, const TENSOR& x, const TENSOR& y, const TENSOR& z){
			       lambda(b,null_ix,x,y,z);});
	return;
      }

      TENSOR _x(x);
      GstridesB xstrides=GstridesB::zero(kg);
      if(x.strides.size()==nc+1+kg){
	xstrides=x.strides.chunk(0,kg);
	_x.dims=Gdims::cat(_x.dims[0],_x.dims.chunk(1+kg));
	_x.strides=Gdims::cat(_x.dims[0],_x.dims.chunk(1+kg));
      }

      TENSOR _y(y);
      GstridesB ystrides=GstridesB::zero(kg);
      if(x.strides.size()==nc+1+kg){
	xstrides=y.strides.chunk(0,kg);
  	_y.dims=Gdims::cat(_y.dims[0],_y.dims.chunk(1+kg));
	_y.strides=Gdims::cat(_y.dims[0],_y.dims.chunk(1+kg));
      }

      TENSOR _z(z);
      GstridesB zstrides=GstridesB::zero(kg);
      if(z.strides.size()==nc+1+kg){
	zstrides=z.strides.chunk(0,kg);
  	_z.dims=Gdims::cat(_z.dims[0],_z.dims.chunk(1+kg));
	_z.strides=Gdims::cat(_z.dims[0],_z.dims.chunk(1+kg));
      }

      for_each_batch_multi(_x,_y,_z,[&](const int b, const TENSOR& x, const TENSOR& y, const TENSOR& z){
			     TENSOR _x(x);
			     TENSOR _y(y);
			     TENSOR _z(z);
			     for(int i=0; i<ncells; i++){
			       Gindex ix(i,gdims);
			       _x.arr=x.arr+xstrides.offs(ix);
			       _y.arr=y.arr+ystrides.offs(ix);
			       _z.arr=z.arr+zstrides.offs(ix);
			       lambda(b,ix,_x,_y,_z);
			     }
			   });
    }


  public: // ---- Products -------------------------------------------------------------------------------------------------

    #include "BGtensor_products.hpp"

    /*
    template<typename TYPE2>
    BGtensor prod(const BGtensor<TYPE2>& y){
      BGtensor<TYPE,nc> R(dominant_batch(*this,y),dominant_gdims(*this,y),get_cdims(),0,get_dev());
      return R;
    }
    */

  public: // ---- I/O ------------------------------------------------------------------------------------------------------


    string cell_to_string(int b, const Gindex& ix, const string indent="") const{
      if(nc==0){
	ostringstream oss;
	oss<<"("<<arr[b*strides[0]+gstrides().offs(ix)]<<")"<<endl;
	return oss.str();
      }
      return cell(b,ix).str(indent);
    }

    string batch_to_string(int b, const string indent="") const{
      ostringstream oss;
      if(ngdims()==0) 
	oss<<cell_to_string(b,Gindex(),indent);
      else{
	Gdims gdims=get_gdims();
	int ncells=gdims.asize();
	for(int i=0; i<ncells; i++){
	  Gindex ix(i,gdims);
	  oss<<indent<<"Cell "<<ix<<": ";
	  if(nc>0) oss<<endl;
	  oss<<cell_to_string(b,ix,indent+"  ");
	}
      }
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      if(getb()==1) 
	oss<<batch_to_string(0,indent);
      else{
	for(int i=0; i<getb(); i++){
	  oss<<indent<<"Batch "<<i<<": ";
	  if(ndims()>1) oss<<endl;
	  oss<<batch_to_string(i,indent+"  ");
	}
      }
      return oss.str();
    }

    string str(const string indent="") const{
      return to_string(indent);
    }

    friend ostream& operator<<(ostream& stream, const BGtensor& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE, typename TYPE2>
  inline BGtensor<TYPE> operator*(const TYPE2 c, const BGtensor<TYPE>& x){
    return x*c;
  }

}


#endif 
