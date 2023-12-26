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


#ifndef _ptr_indexed_object_bank
#define _ptr_indexed_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY, typename OBJ>
  class ptr_indexed_object_bank: public unordered_map<KEY*,OBJ>{
  public:

    using unordered_map<KEY*,OBJ>::insert;
    using unordered_map<KEY*,OBJ>::find;
    using unordered_map<KEY*,OBJ>::erase;


    std::function<OBJ(const KEY&)> make_obj;
    observer<KEY> observers;
    
    ~ptr_indexed_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_indexed_object_bank():
      make_obj([](const KEY& x){cout<<"empty object in bank"<<endl; return OBJ();}),
      observers([this](KEY* p){erase(p);}){}

    ptr_indexed_object_bank(std::function<OBJ(const KEY&)> _make_obj):
      make_obj(_make_obj),
      observers([this](KEY* p){erase(p);}){}


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY& key){
      return (*this)(&key);
    }

    OBJ operator()(const KEY& key){
      return (*this)(&const_cast<KEY&>(key));
    }

    OBJ& operator()(KEY* keyp){
      cout<<keyp<<endl;
      auto it=find(keyp);
      if(it!=unordered_map<KEY*,OBJ>::end()) 
	return it->second;
      cout<<"make"<<endl;
      observers.add(keyp);
      auto p=insert({keyp,make_obj(*keyp)});
      return p.first->second;
    }

  };



}

#endif 


  /*
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
      make_obj([](const KEY& x){cout<<"shared_obj_bank error"<<endl; return nullptr;}),
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

      auto new_obj=shared_ptr<OBJ>(make_obj(*keyp));
      (*this)[keyp]=new_obj;
      observer.add(keyp);
      return new_obj;
    }

  };
  */
