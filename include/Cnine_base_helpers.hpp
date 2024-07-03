
// ---- Helper functions -----------------------------------------------------------------------------------


namespace cnine{

  inline int roundup(const int x, const int s){
    return ((x-1)/s+1)*s;
  }

  inline int roundup(const int x){
    return ((x-1)/32+1)*32;
  }

  template<typename TYPE>
  inline TYPE ifthen(const bool p, const TYPE& x, const TYPE& y){
    if(p) return x; else return y;
  }

  template<typename TYPE>
  inline TYPE bump(TYPE& x, TYPE y){
    if(y>x) x=y;
    return x;
  }

  template<typename TYPE>
  inline void fastadd(const TYPE* source, TYPE* dest, const int n){
    for(int i=0; i<n; i++)
      *(dest+i)+=*(source+i);
  }

  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=beg[i];
  }

  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest, TYPE c){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=c*beg[i];
  }

  template<typename TYPE>
  void stdsub(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]-=beg[i];
  }

  template<typename TYPE1, typename TYPE2>
  inline std::vector<TYPE1> convert(std::vector<TYPE2>& x){
    std::vector<TYPE1> R(x.size());
    for(int i=0; i<x.size(); i++)
      R[i]=TYPE1(x[i]);
    return R;
  }

  template<typename TYPE1, typename TYPE2>
  inline std::vector<TYPE2> mapcar(const std::vector<TYPE1>& v, 
    const std::function<TYPE2(const TYPE1&)> lambda){
    std::vector<TYPE2> R;
    for(auto& p: v)
      R.push_back(lambda(p));
    return R;
  }


  // ---- Printing -------------------------------------------------------------------------------------------


  template<typename TYPE>
  std::ostream& print(const TYPE& x){
    cout<<x.str()<<endl;
    return cout;
  }

  template<typename TYPE>
  inline ostream& print(const string name, const TYPE& x){
    cout<<name<<"="<<x.str()<<endl;
    return cout; 
  }

  template<typename TYPE>
  inline ostream& printl(const string name, const TYPE& x){
    cout<<name<<"="<<endl<<x.str()<<endl;
    return cout; 
  }

  inline ostream& operator<<(ostream& stream, const vector<int>& v){
    stream<<"(";
    int I=v.size()-1;
    for(int i=0; i<I; i++)
      stream<<v[i]<<",";
    if(v.size()>0) 
      stream<<v[v.size()-1];
    stream<<")";
    return stream;
  }

  extern string base_indent;

  struct indenter{
  public:

    string old;

    indenter(const string s){
      old=base_indent;
      base_indent=base_indent+s;
    }

    ~indenter(){
      base_indent=old;
    }

  };

}

#define PRINTL(x) printl(#x,x);
