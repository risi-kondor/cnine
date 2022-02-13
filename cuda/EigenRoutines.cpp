//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2022, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineEigenRoutines
#define _CnineEigenRoutines

#include <Eigen/Dense>

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor2_view.hpp"
#include "RtensorObj.hpp"


namespace cnine{


  pair<RtensorObj,RtensorObj> eigen_eigendecomp(const Rtensor2_view& x){
    int d=x.n0;
    assert(d==x.n1);
    Eigen::MatrixXd M(d,d);
    for(int i=0; i<d; i++) for(int j=0; j<d; j++) M(i,j)=x(i,j);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    auto Ue=solver.eigenvectors();
    RtensorObj U(dims(d,d));
    for(int i=0; i<d; i++) for(int j=0; j<d; j++) U(i,j)=Ue(i,j);
    auto De=solver.eigenvalues();
    RtensorObj D(dims(x.n0));
    for(int i=0; i<d; i++) D(i)=De(i);
    return make_pair(U,D);
  }


  RtensorObj eigen_linsolve(const Rtensor2_view& _A, const Rtensor1_view& _b){
    int I=_A.n0;
    int J=_A.n1;
    assert(I==_b.n0);

    Eigen::MatrixXd A(I,J);
    for(int i=0; i<I; i++) for(int j=0; j<J; j++) A(i,j)=_A(i,j);
    Eigen::VectorXd b(I);
    for(int i=0; i<I; i++) b(i)=_b(i);

    RtensorObj _x(dims(J),fill_raw());
    auto x=A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    for(int i=0; i<J; i++) _x(i)=x(i);

    return _x;
  }

}

#endif 
