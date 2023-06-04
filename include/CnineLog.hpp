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


#ifndef _CnineLog
#define _CnineLog

#include <fstream>
#include <chrono>
#include <ctime>

namespace cnine{

  class CnineLog{
  public:

    ofstream ofs;

    chrono::time_point<chrono::system_clock> topen;

    CnineLog(const string filename="cnine.log"){
      ofs.open(filename,std::ios_base::app);
      topen = std::chrono::system_clock::now();
      std::time_t timet = std::chrono::system_clock::to_time_t(topen);
      ofs<<"Cnine log opened on "<<std::ctime(&timet)<<endl;

    }

    ~CnineLog(){
      auto elapsed=chrono::duration<double>(chrono::system_clock::now()-topen).count();
      ofs<<endl<<"Cnine log closed after "<<to_string(elapsed)<<" seconds."<<endl;
      ofs<<"----------------------------------------------------------------------"<<endl<<endl;
      ofs.close();
    }


  public: // -------------------------------------------------------------------------------------------------


    void operator()(const string msg){
      ofs<<msg<<endl;
    }

    void error(const string fn, const string str){
      ofs<<"Error in "<<fn<<": "<<str<<endl;
      CoutLock lk;
      cout<<"Error in "<<fn<<": "<<str<<endl;
    }

    void warning(const string fn, const string str){
      ofs<<"Warning in "<<fn<<": "<<str<<endl;
      CoutLock lk;
      cout<<"Warning in "<<fn<<": "<<str<<endl;
    }


    //void operator()(const string msg){
    //std::time_t timet = std::time(nullptr);
    //char os[30];
    //strftime(os,30,"%H:%M:%S ",std::localtime(&timet));
    //ofs<<os<<msg<<endl;
    //}

    void operator()(){}

  };



}

#endif 
