
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _ThreadGroup
#define _ThreadGroup

#include <thread>


namespace cnine{

  class ThreadGroup{
  public:

    int maxthreads=4;
    mutex mx;
    condition_variable start_thread_cv;
    mutex start_thread_mx;
    atomic<int> nrunning;
    vector<thread> threads;


  public:


    ThreadGroup()=delete;
  
    ThreadGroup(const int _maxthreads=4): 
      maxthreads(_maxthreads){
      nrunning=0;
    }

    ~ThreadGroup(){
      for(auto& p:threads)
	p.join();
    }


  public:

    template<typename FUNCTION, typename OBJ>
    void add(const int nsubthreads, FUNCTION lambda, OBJ arg0){
      lock_guard<mutex> guard(mx);

      if(nrunning>=maxthreads){
	unique_lock<mutex> lock(start_thread_mx);
	start_thread_cv.wait(lock,[this](){return nrunning<maxthreads;});
      }

      nrunning++;
      threads.push_back(thread([this,nsubthreads,lambda,arg0](){
	    nthreads=nsubthreads;
	    lambda(arg0);
	    this->done();
	  }));
    }	 


    void done(){
      nrunning--;
      start_thread_cv.notify_one();
    }

  };


}

#endif 
