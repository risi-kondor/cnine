#ifndef _TensorAccessorBLAS
#define _TensorAccessorBLAS

#include <cblas.h>

#include "TensorAccessor.hpp"


namespace cnine{


  class TensorLayout{
  public:
    
    vector<int> indices;
    vector<int> stride_order;


  };
  
  
  class GEMMgenerator{
  public:

    bool ready=false;

    GEMMgenerator(const TensorLayout& rlayout, const TensorLayout& xlayout, const TensorLayout& ylayout){
      ready=build_generatr(rlayout,xlayout,ylayout);
    }

    build_generator(const TensorLayout& rlayout, const TensorLayout& xlayout, const TensorLayout& ylayout){
      
    }


  };


  //inline void add_gemm(const TensorAccessor<float,2>& r, 
  //const TensorAccessor<float,2>& x, const TensorAccessor<float,2>& y){
  //}


}


#endif 
