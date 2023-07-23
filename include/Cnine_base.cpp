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


#include "Cnine_base.hpp"
#include "Primes.hpp"
#include "Factorial.hpp"
#include "FFactorial.hpp"
#include "DeltaFactor.hpp"
#include "CnineLog.hpp"

#ifdef _WITH_CENGINE
#include "Cengine_base.cpp"
#endif 

namespace cnine{

  thread_local int nthreads=1;
  float* cuda_oneS=nullptr;

  int streaming_footprint=1024;
  thread_local DeviceSelector dev_selector;

  Primes primes;
  Factorial factorial;
  FFactorial ffactorial;
  DeltaFactor delta_factor;

  CnineLog cnine_log;

}

std::default_random_engine rndGen;

mutex cnine::CoutLock::mx;


#ifdef _WITH_CENGINE
#include "RscalarM.hpp"
#include "CscalarM.hpp"
#include "CtensorM.hpp"

/*
int Cengine::ctensor_add_op::_batcher_id=0; 
 int Cengine::ctensor_add_op::_rbatcher_id=0; 

 int Cengine::ctensor_add_prod_c_A_op::_rbatcher_id=0; 

 int Cengine::ctensor_add_inp_op::_rbatcher_id=0; 

*/

namespace cnine{


  template<> int ctensor_add_Mprod_op<0,0>::_batcher_id=0; 
  template<> int ctensor_add_Mprod_op<0,1>::_batcher_id=0; 
  template<> int ctensor_add_Mprod_op<0,2>::_batcher_id=0; 
  template<> int ctensor_add_Mprod_op<0,3>::_batcher_id=0; 
  
 template<> int ctensor_add_Mprod_op<1,0>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,1>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,2>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,3>::_batcher_id=0; 

 template<> int ctensor_add_Mprod_op<2,0>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,1>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,2>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,3>::_batcher_id=0; 

 template<> int ctensor_add_Mprod_op<0,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,3>::_rbatcher_id=0; 

 template<> int ctensor_add_Mprod_op<1,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,3>::_rbatcher_id=0; 

 template<> int ctensor_add_Mprod_op<2,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,3>::_rbatcher_id=0; 

}

#endif
