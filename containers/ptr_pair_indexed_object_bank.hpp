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


#ifndef _ptr_pair_indexed_object_bank
#define _ptr_pair_indexed_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY0, typename KEY1, typename OBJ>
  class ptr_pair_indexed_object_bank: public unordered_map<std::pair<KEY0*,KEY1*>,OBJ>{
  public:

    typedef std::pair<KEY0*,KEY1*> KEYS;

    using unordered_map<KEYS,OBJ>::insert;
    using unordered_map<KEYS,OBJ>::find;
    using unordered_map<KEYS,OBJ>::erase;


    std::function<OBJ(const KEY0&, const KEY1&)> make_obj;
    observer<KEY0> observers0;
    observer<KEY1> observers1;
    //map_of_lists<KEY0*,KEY1*> lookup0;
    //map_of_lists<KEY1*,KEY0*> lookup1;
    std::unordered_map<KEY0*,std::set<KEY1*> > lookup0;
    std::unordered_map<KEY1*,std::set<KEY0*> > lookup1;

    ~ptr_pair_indexed_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_pair_indexed_object_bank():
      make_obj([](const KEY0& x0, const KEY1& x1){cout<<"empty object in bank"<<endl; return OBJ();}),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}){}

    ptr_pair_indexed_object_bank(std::function<OBJ(const KEY0&, const KEY1&)> _make_obj):
      make_obj(_make_obj),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}){}


  private:

    void erase0(KEY0* x){
      for(auto y:lookup0[x]){
	erase(make_pair(x,y));
	lookup1[y].erase(x);
      }
    }

    void erase1(KEY1* y){
      for(auto x:lookup1[y]){
	erase(make_pair(x,y));
	lookup0[x].erase(y);
      }
    }

  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY0& key0, KEY1& key1){
      return (*this)(&key0,&key1);
    }

    OBJ operator()(const KEY0& key0, const KEY1& key1){
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1));
    }

    OBJ operator()(shared_ptr<KEY0> keyp0, shared_ptr<KEY1> keyp1){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      return (*this)(&key0,&key1);
    }

    OBJ operator()(shared_ptr<const KEY0> keyp0, shared_ptr<const KEY1> keyp1){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1));
    }

    OBJ& operator()(KEY0* keyp0, KEY1* keyp1){
      auto it=find(make_pair(keyp0,keyp1));
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      observers0.add(keyp0);
      observers1.add(keyp1);
      lookup0[keyp0].insert(keyp1);
      lookup1[keyp1].insert(keyp0);
      auto p=insert({make_pair(keyp0,keyp1),make_obj(*keyp0,*keyp1)});
      return p.first->second;
    }

  };



}

namespace std{

  template<typename IX1, typename IX2>
  struct hash<pair<IX1,IX2> >{
  public:
    size_t operator()(const pair<IX1,IX2>& x) const{
      size_t h=hash<IX1>()(x.first);
      h=(h<<1)^hash<IX2>()(x.second);
      return h;
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
