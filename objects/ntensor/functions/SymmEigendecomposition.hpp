/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _SymmEigendecomposition
#define _SymmEigendecomposition

//#include "Tensor.hpp"

#ifdef _WITH_EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>
#endif


namespace cnine{

  template<typename TYPE>
  class SymmEigendecomposition{
  public:

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver;

    SymmEigendecomposition(const TensorView<TYPE>& _A){
      //solver(_A){
      solver.compute(Eigen::MatrixXf(_A));
      //solver.compute(_A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    }

    Tensor<TYPE> U() const{
      return solver.eigenvectors();
    }

    Tensor<TYPE> lambda() const{
      return solver.eigenvalues();
    }


  };

}

#endif 
