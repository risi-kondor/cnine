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


#ifndef _op_object_bank
#define _op_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY, typename OBJ>
  class shared_object_bank: public unordered_map<KEY*,shared_ptr<OBJ> >{
  public:

    using unordered_map<KEY*,shared_ptr<OBJ> >::find;
    using unordered_map<KEY*,shared_ptr<OBJ> >::erase;


    std::function<OBJ*(const KEY&)> make_obj;
    observer<KEY> observer;
    
    ~shared_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    shared_object_bank():
      make_obj([](const KEY& x){return nullptr;}),
      observer([this](KEY* p){erase(p);}){}

    shared_object_bank(std::function<OBJ*(const KEY&)> _make_obj):
      make_obj(_make_obj),
      observer([this](KEY* p){erase(p);}){}


  public: // ---- Access -------------------------------------------------------------------------------------


    shared_ptr<OBJ> operator()(KEY& key){
      return (*this)(&key);
    }

    shared_ptr<OBJ> operator()(const KEY& key){
      return (*this)(&const_cast<KEY&>(key));
    }

    shared_ptr<OBJ> operator()(KEY* keyp){
      auto it=find(keyp);
      if(it!=unordered_map<KEY*, shared_ptr<OBJ> >::end()) 
	return it->second;

      OBJ* new_obj=make_obj(*keyp);
      (*this)[keyp]=shared_ptr<OBJ>(new_obj);
      observer.add(keyp);
      return shared_ptr<OBJ>(new_obj);
    }

  };

}

#endif 
