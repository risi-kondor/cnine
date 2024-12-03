#ifndef _CnineTikzStream
#define _CnineTikzStream

namespace cnine{

  class TikzStream{
  public:

    int depth=0;
    ostringstream oss;

    void add_line(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    void write(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    void operator<<(const string& s){
      oss<<s;
    }

    string str(){
      return oss.str();
    }

  };

}

#endif 
