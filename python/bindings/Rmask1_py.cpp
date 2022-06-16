//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


pybind11::class_<Rmask1>(m,"Rmask1")

  .def(pybind11::init([](at::Tensor& M){
      return Rmask1::matrix(RtensorA::view(M).view2());
      }))

  .def("inv",&Rmask1::inv)

  .def("__str__",&Rmask1::str,py::arg("indent")="")
  
  ;
