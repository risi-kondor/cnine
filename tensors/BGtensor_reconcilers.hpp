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

#ifndef _BGtensor_reconcilers
#define _BGtensor_reconcilers

namespace cnine{

  template<typename TYPE>
  class BGtensor;


  // ---- Batches -------------------------------------------------------------------------------------------


  template<typename TYPE1, typename TYPE2>
  static int dominant_batch(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y){
    int xb=x.getb();
    int yb=y.getb();
    if(xb==yb) return xb;
    if(xb==1) return yb;
    if(yb==1) return xb;
    throw std::invalid_argument("Cnine error: the batch dimensions of "+x.repr()+" and "+y.repr()+
      " cannot be reconciled.");
    return 0;
  }


  template<typename TYPE1, typename TYPE2, typename TYPE3>
  static int dominant_batch(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y, const BGtensor<TYPE3>& z){
    int xb=x.getb();
    int yb=y.getb();
    int zb=z.getb();
    CNINE_ASSRT(xb>0 && yb>0 && zb>0);

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


  template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
  static int dominant_batch(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y, const BGtensor<TYPE3>& z, const BGtensor<TYPE4>& w){
    int xb=x.getb();
    int yb=y.getb();
    int zb=z.getb();
    int wb=w.getb();
    CNINE_ASSRT(xb>0 && yb>0 && zb>0 && wb>0);

    int B=1;
    if(xb>1) B=xb;
    if(yb>1) B=yb;
    if(zb>1) B=zb;
    if(wb>1) B=wb;
    CNINE_ASSRT(xb==B || xb==1);
    CNINE_ASSRT(yb==B || yb==1);
    CNINE_ASSRT(zb==B || zb==1);
    CNINE_ASSRT(wb==B || wb==1);

    return B;
  }


  // ---- Grid ----------------------------------------------------------------------------------------------


  template<typename TYPE1, typename TYPE2>
  Gdims dominant_gdims(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y){
    Gdims xg=x.gdims();
    Gdims yg=y.gdims();
    if(xg==yg) return xg;
    if(!x.is_grid()) return yg;
    if(!y.is_grid()) return xg;
    throw std::invalid_argument("Genet error: the grid dimensions of "+x.repr()+" and "+y.repr()+
      " cannot be reconciled.");
    return Gdims();
  }

  template<typename TYPE1, typename TYPE2, typename TYPE3>
  Gdims dominant_gdims(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y, const BGtensor<TYPE3>& z){
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


  template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
  Gdims dominant_gdims(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y, const BGtensor<TYPE3>& z, const BGtensor<TYPE4>& w){
    Gdims xg=x.gdims();
    Gdims yg=y.gdims();
    Gdims zg=y.gdims();
    Gdims wg=w.gdims();

    Gdims r;
    if(xg.size()>0) r=xg;
    if(yg.size()>0) r=yg;
    if(zg.size()>0) r=zg;
    if(wg.size()>0) r=wg;
    CNINE_ASSRT(xg.size()==0 || xg==r);
    CNINE_ASSRT(yg.size()==0 || yg==r);
    CNINE_ASSRT(zg.size()==0 || zg==r);
    CNINE_ASSRT(wg.size()==0 || wg==r);
    return r;
  }


  // ---- Cells ---------------------------------------------------------------------------------------------


  template<typename TYPE1, typename TYPE2>
  static Gdims dominant_cdims(const BGtensor<TYPE1>& x, const BGtensor<TYPE2>& y){
    Gdims xg=x.get_cdims();
    Gdims yg=y.get_cdims();
    if(xg==yg) return xg;
    if(!x.has_cells()) return yg;
    if(!y.has_cells()) return xg;
    throw std::invalid_argument("Genet error: the cell dimensions of "+x.repr()+" and "+y.repr()+" cannot be reconciled.");
    return Gdims();
  }

}

#endif 
