#ifndef _JSONlike
#define _JSONlike

namespace cnine{


  class JSONlike{
  public:

    class JSONlikeHandle; 

    string val;
    map<string,shared_ptr<JSONlike> > dict;
    vector<shared_ptr<JSONlike> > vec;
    bool valid=true;

    JSONlike(){}

    //JSONlike(const string _val):
    //val(_val){}

    template<typename TYPE>
    JSONlike(const vector<TYPE>& x){set(x);}

    template<typename TYPE>
    JSONlike(const initializer_list<TYPE>& x){
      ostringstream oss;
      oss<<"[";
      bool first=true;
      for(auto& p: x){
	cout<<p<<endl;
	if(!first) oss<<","; else first=false;
	oss<<to_string(p);
      }
      oss<<"]";
      val=oss.str();
    }


  public: // ---- Access --------------------------------------------------------------------------------------------


    bool is_set(const string key) const{
      return dict.find(key)!=dict.end();
    }

    JSONlikeHandle operator[](const string key){
      auto it=dict.find(key);
      if(it!=dict.end()) return JSONlikeHandle(it->second);
      auto r=make_shared<JSONlike>();
      dict[key]=r;
      return JSONlikeHandle(r);
    }

    void set(const int x){
      set(to_string(x));
    }

    void set(const string _val){
      val=string("\""+_val+"\"");
    }

    template<typename TYPE>
    void set(const initializer_list<TYPE>& x){
      ostringstream oss;
      oss<<"[";
      bool first=true;
      for(auto& p: x){
	if(!first) oss<<","; else first=false;
	oss<<to_string(p);
      }
      oss<<"]";
      val=oss.str();
    }
    
    template<typename TYPE, class = decltype(to_string(std::declval<const TYPE&>()))>
    void set(const vector<TYPE>& x){
      ostringstream oss;
      oss<<"[";
      bool first=true;
      for(auto& p: x){
	if(!first) oss<<","; else first=false;
	oss<<to_string(p);
      }
      oss<<"]";
      val=oss.str();
    }

    template<typename TYPE>
    void set(const vector<vector<TYPE> >& x){
      ostringstream oss;
      oss<<"[";
      bool first=true;
      for(auto& p: x){
	if(!first) oss<<","; else first=false;
	oss<<"[";
	bool first2=true;
	for(auto& q: p){
	  if(!first2) oss<<","; else first2=false;
	  oss<<to_string(q);
	}
	oss<<"]";
      }
      oss<<"]";
      val=oss.str();
    }

    int get_int() const{
      return stoi(get_str());
    }

    string get_str() const{
      if(val.size()>1 && val[0]=='"' && val.back()=='"')
	return string(val.begin()+1,val.end()-1);
      return val;
    }

    template<typename TYPE>
    vector<TYPE> get_vec() const{
      return get_vec<TYPE>(val);
    }

    template<typename TYPE>
    vector<vector<TYPE> > get_vec_vec() const{
      vector<vector<TYPE> > R;
      auto S=get_vec<string>(val);
      for(auto& p: S)
	R.push_back(get_vec<TYPE>(p));
      return R;
    }

    template<typename TYPE>
    vector<TYPE> get_vec(string x) const{
      vector<TYPE> R;
      if(x.size()<2 || x[0]!='[') return R;
      int s=1;
      do{
	auto r=end_of_term(x,s);
	if(r==string::npos) r=x.size()-1;
	std::stringstream ss(x.substr(s,r-s));
	TYPE v;
	ss>>v;
	R.push_back(v);
	s=x.find_first_not_of(" \n\t\r,]}",r);
      }while(s!=string::npos);
      return R;
    }


  public: // ---- Handle ---------------------------------------------------------------------------------------------


    class JSONlikeHandle{
    public:

      shared_ptr<JSONlike> obj;
      JSONlikeHandle(const shared_ptr<JSONlike>& _obj):
	obj(_obj){}

      void operator=(const int x){
	obj->set(x);
      }

      void operator=(const string _val){
	obj->set(_val);
      }

      template<typename TYPE>
      void operator=(const initializer_list<TYPE>& x){
	obj->set(x);
      }

      template<typename TYPE>
      void operator=(const vector<TYPE>& x){
	obj->set(x);
      }

      operator int() const{
	return obj->get_int();
      }

      int get_int() const{
	return obj->get_int();
      }

      operator string() const{
	return obj->get_str();
      }

      string get_str() const{
	return obj->get_str();
      }

      template<typename TYPE>
      vector<TYPE> get_vec() const{
	return obj->get_vec<TYPE>();
      }

      template<typename TYPE>
      vector<vector<TYPE> > get_vec_vec() const{
	return obj->get_vec_vec<TYPE>();
      }

    };


  public: // ---- Parsing --------------------------------------------------------------------------------------------


    JSONlike(const string str){
      valid=false;
      auto s=str.find_first_not_of(" \n\t\r");
      if(s==string::npos) return;
      if(str[s]=='{'){
	auto e=str.find("}",s);
	if(e==string::npos) return;
	parse_dict(str.substr(s+1,e-s-1));
	return;
      }
      auto e=str.find_first_of(" \n\t\r");
      val=str.substr(s,e-s);
      valid=true;
    }
    
    void parse_dict(const string str){
      auto s=str.find_first_not_of(" \n\t\r");
      while(s!=string::npos && s<str.size() && dict.size()<10){
	auto e=str.find_first_of(" =\n\t",s);
	if(e==string::npos) return;
	auto key=str.substr(s,e-s);
	//cout<<"key="<<key<<endl;
	auto eq=str.find("=",e);
	if(eq==string::npos) return;
	auto f=str.find_first_not_of(" \n\t\r",eq+1);
	if(f==string::npos) return;
	auto g=end_of_term(str,f);
	auto v=str.substr(f,g-f);
	dict[key]=make_shared<JSONlike>(v);
	if(g==string::npos) break;
	s=str.find_first_not_of(" \n\t\r,}",g);
      }
      valid=true;
    }

    string::size_type end_of_term(const string str, string::size_type s) const{
      if(str[s]=='"'){
	auto r=str.find("\"",s+1);
	if(r!=string::npos && r<str.size()-1) return r+1; else return string::npos;
      }
      auto r=str.find_first_of(" \n\t\r,}[",s);
      while(r!=string::npos && str[r]=='['){
	auto s=end_of_clause(str,r,']');
	r=str.find_first_of(" \n\t\t,}[{",s);
      }
      return r;
    }

    string::size_type end_of_clause(const string str, string::size_type s, char c) const{
      auto r=str.find_first_of("][",s+1);
      while(r!=string::npos && str[r]=='['){
	auto s=end_of_clause(str,r,']');
	r=str.find_first_of("][",s+1);
      }
      return r;
    }


  public: // ---- Output ---------------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      if(dict.size()>0){
	oss<<"{";
	bool first=true;
	for(auto& p: dict){
	  if(!first){oss<<",";} else first=false;
	  //oss<<"\""<<p.first<<"\"="<<p.second->str();
	  oss<<p.first<<"="<<p.second->str();
	}
	oss<<"}";
	return oss.str();
      }
      if(vec.size()>0){
	oss<<"[";
	bool first=true;
	for(auto& p: vec){
	  if(!first){oss<<",";} else first=false;
	  oss<<p->str();
	}
      	oss<<"}";
	return oss.str();
      }
      oss<<val;
      return oss.str();
    }

    string str(string indent) const{
      ostringstream oss;
      if(dict.size()>0){
	oss<<indent<<"{";
	bool first=true;
	for(auto& p: dict){
	  if(!first){oss<<",";} else first=false;
	  //	  oss<<endl<<indent<<"  \""<<p.first<<"\"="<<p.second->str(indent+"  ");
	  oss<<endl<<indent<<"  "<<p.first<<"="<<p.second->str(indent+"  ");
	}
	oss<<endl<<indent<<"}";
	return oss.str();
      }
      if(vec.size()>0){
	oss<<indent<<"[";
	bool first=true;
	for(auto& p: vec){
	  if(!first){oss<<",";} else first=false;
	  oss<<endl<<indent<<"  "<<p->str(indent+"  ");
	}
      	oss<<endl<<indent<<"}";
	return oss.str();
      }
      oss<<val;
      return oss.str();
    }

    string lstr(string indent="") const{
      ostringstream oss;
      oss<<indent<<"JSONlike{"<<endl;
      do{
	if(dict.size()>0){
	  for(auto& p:dict){
	    oss<<indent<<"  "<<p.first<<"=";
	    if(p.second->dict.size()==0)
	      oss<<p.second->val<<endl;
	    else 
	      oss<<endl<<p.second->lstr(indent+"  ");
	  }
	  break;
	}
	oss<<indent<<"  "<<val;
      }while(false);
      oss<<indent<<"}"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& oss, const JSONlike& x){
      oss<<x.str(); return oss;}

  };

  template<typename TYPE> 
  TYPE get_if_set(const JSONlike& x, const string key){
    if(!x.is_set(key)){TYPE r; return r;}
    TYPE r=const_cast<JSONlike&>(x)[key];
    return r;
  }

}

#endif 
