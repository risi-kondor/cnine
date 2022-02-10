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

}

#endif 
