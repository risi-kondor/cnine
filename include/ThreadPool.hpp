
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _ThreadPool
#define _ThreadPool

namespace cnine{

  class ThreadPool{
  public:

    mutex mx;
    mutex gate;
    atomic<int> nthreads;
    int maxthreads=4;
    vector<thread> threads;


  public:

    ThreadPool()=delete;
  
    ThreadPool(const int _maxthreads=4): 
      maxthreads(_maxthreads){}

    ~ThreadPool(){
      while(nrunning>0){
	running_cv.wait();
      }
    }


  public:

    template<class FUNCTION>
    launch(FUNCTION lambda){
      
    }


    template<class FUNCTION, class OBJ>
    inline void ThreadPool::add(FUNCTION lambda, const OBJ x){
      lock_guard<mutex> lock(mx); //                                   unnecessary if called from a single thread
      threadManager.enqueue(this);
      gate.lock(); //                                                  gate can only be unlocked by threadManager
      nthreads++;
      threads.push_back(thread([this,lambda](OBJ _x){
	    lambda(_x); 
	    nthreads--; threadManager.release(this);},x));
      #ifdef _THREADBANKVERBOSE
      printinfo();
      #endif
 }



    bool is_ready(){return nthreads<maxthreads;}

    void printinfo();

  
  };


}


#endif 
