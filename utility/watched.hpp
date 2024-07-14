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


#ifndef _cnine_watched
#define _cnine_watched

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{

  template<typename OBJ> 
  class watcher; 


  template<typename OBJ> 
  class watched: public observable<watched<OBJ> >{
  public:


    watcher<OBJ>& _watcher;

    shared_ptr<OBJ> obj;

    std::function<shared_ptr<OBJ>()> make_obj;

    //watched():
    //make_obj([](){return OBJ();}){}

    watched(watcher<OBJ>& __watcher, std::function<shared_ptr<OBJ>()> _make_obj):
      observable<watched<OBJ> >(this),
      _watcher(__watcher),
      make_obj(_make_obj){}

    ~watched(){
    }


  public: // ---- Access -------------------


    operator OBJ&(){
      return (*this)();
    }

    OBJ& operator()(){
      if(obj.get()==nullptr) make();
      return *obj;
    }

    shared_ptr<OBJ> shared(){
      if(obj.get()==nullptr) make();
      return obj;
    }

    void make(){
      obj=make_obj();
      _watcher.add(this);
    }

    string str() const{
      if(obj.get()==nullptr) return "";
      return ""; //to_string(*obj);
    }

  };



  template<typename OBJ>
  class watcher{
  public:

    observer<watched<OBJ> > _watched;

    void add(watched<OBJ>* x){
      _watched.add(x);
    }

    string str() const{
      ostringstream oss;
      oss<<"[";
      for(auto p:_watched.targets)
	oss<<p->str()<<",";
      oss<<"]";
      return oss.str();
   }

    friend ostream& operator<<(ostream& stream, const watcher& x){
      stream<<x.str(); return stream;
    }

  };



}


#endif 
