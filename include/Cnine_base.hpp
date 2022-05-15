// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Cnine_base
#define _Cnine_base

#include <assert.h>

#include <complex>
#include <iostream>
#include <unordered_map>
#include <random>
#include <functional> 
#include <thread>
#include <mutex>
#include <array>
#include <set>
#include <list>

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 


using namespace std; 

#define CNINE_ASSERT(condition, message) \
  if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }

#define CNINE_UNIMPL() printf("Cnine error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);

#ifdef CNINE_COPY_WARNINGS
#define CNINE_COPY_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" copied."<<endl;
#else 
#define CNINE_COPY_WARNING()
#endif 

#ifdef CNINE_ASSIGN_WARNINGS
#define CNINE_ASSIGN_WARNING() cout<<"\e[1mWarning:\e[0m "<<classname()<<" assigned."<<endl;
#else
#define CNINE_ASSIGN_WARNING() 
#endif

#ifdef CNINE_MOVE_WARNINGS
#define CNINE_MOVE_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" moved."<<endl;
#else 
#define CNINE_MOVE_WARNING()
#endif 

#ifdef CNINE_MOVEASSIGN_WARNINGS
#define CNINE_MOVEASSIGN_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" move assigned."<<endl;
#else 
#define CNINE_MOVEASSIGN_WARNING()
#endif 

#ifdef CNINE_ATEN_CONVERT_WARNINGS
#define CNINE_CONVERT_FROM_ATEN_WARNING() cout<<"\e[1mcnine:\e[0m ATen tensor converted to "<<classname()<<"."<<endl;
#define CNINE_CONVERT_TO_ATEN_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" converted to ATen tensor."<<endl;
#else 
#define CNINE_CONVERT_FROM_ATEN_WARNING()
#define CNINE_CONVERT_TO_ATEN_WARNING()
#endif 

#ifdef CNINE_RANGE_CHECKING
#define CNINE_CHECK_RANGE(expr) expr
#else
#define CNINE_CHECK_RANGE(expr)
#endif

#ifdef CNINE_SIZE_CHECKING
#define CNINE_CHECK_SIZE(expr) expr
#else
#define CNINE_CHECK_SIZE(expr)
#endif

#ifdef CNINE_DEVICE_CHECKING
#define CNINE_CHECK_DEV(expr) expr
#else
#define CNINE_CHECK_DEV(expr)
#endif


#define CNINE_NOCUDA_ERROR cout<<"Error: Cnine was compiled without GPU support."<<endl;
#define CNINE_CPUONLY() if(dev!=0) {printf("Cnine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}

#define COUT(cmd) {cnine::CoutLock lk; cout<<cmd<<endl;}
#define CNINE_COUT(cmd) {CoutLock lk; cout<<cmd<<endl;}

#define CNINE_CHECK_DIMS(a,b,fn) if(a!=b) {{CoutLock lk; cerr<<"cnine error in function "<<fn<<": dimension mismatch."<<endl;} exit(1);}
#define CNINE_CHECK_DIMS2(a,b,c,fn) if(a!=b||a!=c) {{CoutLock lk; cerr<<"cnine error in function "<<fn<<": dimension mismatch."<<endl;} exit(1);}

#define CNINE_CHECK_BATCH2(x,y) if(x.n0!=y.n0) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": batch dimension mismatch.");
#define CNINE_CHECK_BATCH3(x,y,z) if(x.n0!=y.n0 || x.n0!=z.n0) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": batch dimension mismatch.");

#define CNINE_CHECK_DEV2(x,y) if(x.dev!=y.dev) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");
#define CNINE_CHECK_DEV3(x,y,z) if(x.dev!=y.dev || x.dev!=z.dev) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");



namespace cnine{


  // ---- Fill -----------------------------------------------------------------------------------------------

  struct fill_pattern{};
  struct fill_noalloc: public fill_pattern {fill_noalloc(){}};
  struct fill_raw: public fill_pattern {fill_raw(){}};
  struct fill_zero: public fill_pattern{fill_zero(){}};
  struct fill_view: public fill_pattern{fill_view(){}};
  struct fill_fn: public fill_pattern{fill_fn(){}};
  struct fill_ones: public fill_pattern{fill_ones(){}};
  struct fill_sequential: public fill_pattern{fill_sequential(){}};
  struct fill_identity: public fill_pattern{fill_identity(){}};
  struct fill_uniform: public fill_pattern{fill_uniform(){}};
  struct fill_tensor: public fill_pattern{fill_tensor(){}};
  struct fill_stack: public fill_pattern{fill_stack(){}};
  struct fill_cat: public fill_pattern{fill_cat(){}};
  struct fill_cgaussian: public fill_pattern{fill_cgaussian(){}};

  struct fill_gaussian: public fill_pattern{
  public:
    float c=1.0;
    fill_gaussian(){}
    explicit fill_gaussian(const float _c): c(_c){}
    fill_gaussian operator()(const float _c) const {return fill_gaussian(_c);}
  };

  struct fill_bernoulli: public fill_pattern{
    double p=0.5;
    fill_bernoulli(){}
    fill_bernoulli(const double _p):p(_p){}
  };
  
  struct fill_symm_bernoulli: public fill_pattern{
    double p=0.5;
    fill_symm_bernoulli(){}
    fill_symm_bernoulli(const double _p):p(_p){}};

  template<typename TYPE> 
  struct fill_const: public fill_pattern{
    TYPE p=0;
    fill_const(){}
    fill_const(const TYPE _p):p(_p){}
  };

  namespace fill{
    static const fill_noalloc noalloc;
    static const fill_raw raw; // 0
    static const fill_zero zero; // 1
    static const fill_view view; // 1
    static const fill_fn fn; 
    static const fill_ones ones; // 2 
    static const fill_sequential sequential; //3 
    static const fill_identity identity; //4 
    static const fill_uniform uniform; //5 
    static const fill_tensor tensor; //5 
    static const fill_bernoulli bernoulli; //6 
    static const fill_symm_bernoulli symm_bernoulli; //7
    static const fill_gaussian gaussian; //8
    static const fill_cgaussian cgaussian;
    static const fill_stack stack;
    static const fill_cat cat;
  }


  struct cmap_flag{};
  struct cmap_set: public cmap_flag {cmap_set(){}};
  struct cmap_add: public cmap_flag {cmap_add(){}};


  // ---- Other flags ---------------------------------------------------------------------------------------


  class view_flag{
  public:
    view_flag(){}
  };

  namespace flag{
    static const view_flag view;
  }

  class nowarn_flag{
  public:
    nowarn_flag(){}
  };

  static const nowarn_flag nowarn;


  // --- DeviGPUces ---------------------------------------------------------------------------------------------


  struct device_id{
    int _id;
    device_id(const int x): _id(x){};
    int id() const {return _id;}
  };

  struct device{
    int _id;
    device(const int x): _id(x){};
    //device& operator=(const int x){
    //_id=x;
    //return *this;
    //}
    int id() const {return _id;}
  };

  namespace deviceid{
    //static device_id CPU(0);
    //static device_id GPU0(1);
    static device CPU(0);
    static device GPU0(1);
  }    


  // ---- Formats -------------------------------------------------------------------------------------------


  enum class pack_format{list,compact};

  inline int toint(const pack_format& x){
    if(x==pack_format::list) return 0;
    if(x==pack_format::compact) return 1;
    return 0;
  }


  // ---- Multithreading ------------------------------------------------------------------------------------


  class CoutLock{
  public:
    CoutLock(): lock(mx){}
    lock_guard<std::mutex> lock;
    static std::mutex mx;
  };


  // ---- Helper classes -------------------------------------------------------------------------------------


  struct size_spec: public fill_pattern{
    const int n;
    size_spec(const int _n): n(_n){}
  };

  template<typename TYPE>
  class _viewof{
  public:
    TYPE& obj;
    _viewof(TYPE& _obj): obj(_obj){}
  };

  template<typename TYPE>
  _viewof<TYPE> viewof(TYPE& obj){
    return _viewof<TYPE>(obj);
  }

  template<typename TYPE>
  class _bind0{
  public:
    const TYPE& obj;
    _bind0(const TYPE& _obj): obj(_obj){}
  };

  template<typename TYPE>
  _bind0<TYPE> bind0(const TYPE& obj){
    return _bind0<TYPE>(obj);
  }

  

  class Printable{
  public:
    virtual string str(const string ident="") const=0;
    friend ostream& operator<<(ostream& stream, const Printable& x){
      stream<<x.str(); return stream;}
  };


  // ---- Helper functions -----------------------------------------------------------------------------------


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
    //stream<<"gg";
    return stream;
  }

  /*
  template<class TYPE>
  int sameb(const TYPE& x, const TYPE& y){
    assert(x.dims(0)==y.dims(0));
    return x.dims(0);
  }
  */

  // --- Variadics -------------------------------------------------------------------------------------------


  template<class TYPE, typename... Args>
  vector<TYPE*> variadic_unroller(TYPE& x, Args&... args){
    vector<TYPE*> argv;
    variadic_unroller_sub(argv, x, args...);
    return argv;}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x, Args&... args){
    argv.push_back(&x);
    variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x){
    argv.push_back(&x);}


  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x){
    argv.push_back(&x);}

  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x, Args&... args){
    argv.push_back(&x);
    const_variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_variadic_unroller(const TYPE& x, Args&... args){
    vector<const TYPE*> argv;
    const_variadic_unroller_sub(argv, x, args...);
    return argv;}


  template<class TYPE, class TYPE2, typename... Args>
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x, Args&... args){
    argv.push_back(&x);
    const_derived_variadic_unroller_sub(argv, args...);
  }

  template<class TYPE, class TYPE2> 
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x){
    argv.push_back(&x);
  }

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_derived_variadic_unroller(Args&... args){
    vector<const TYPE*> argv;
    const_derived_variadic_unroller_sub<TYPE>(argv, args...);
    return argv;
  }

}

// ---- CUDA STUFF ------------------------------------------------------------------------------------------


#define CNINE_CONST_MEM_SIZE 32278


#ifdef _WITH_CUDA
#define IFCUDA(cmds) cmds 
#else 
#define IFCUDA(cmds) 
#endif


#ifdef _WITH_CUDA
#define CNINE_REQUIRES_CUDA() 
#else
#define CNINE_REQUIRES_CUDA() printf("Cnine error in \"%s\":  cnine was compiled without CUDA.\n",__PRETTY_FUNCTION__);
#endif 

#ifdef _WITH_CUDA
#define CUDA_SAFE(err) __cudaSafeCall(err, __FILE__, __LINE__ );
inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  if(cudaSuccess!=err){
    fprintf(stderr,"cudaSafeCall() failed at %s:%i : %s\n",file,line,cudaGetErrorString(err));
    exit(-1);}
  return;
}
#else 
#define CUDA_SAFE(err) ; 
#endif 


#ifdef _WITH_CUBLAS
#define CUBLAS_SAFE(expression) {			     \
    cublasStatus_t status= (expression);		     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "CuBLAS error on line " << __LINE__ << ": ";		\
    if(status==CUBLAS_STATUS_SUCCESS) fprintf(stderr,"CUBLAS SUCCESS"); \
    else if(status==CUBLAS_STATUS_NOT_INITIALIZED) \
        fprintf(stderr,"'CUBLAS_STATUS_NOT_INITIALIZED'"); \
    else if(status==CUBLAS_STATUS_ALLOC_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_ALLOC_FAILED'");\
    else if(status==CUBLAS_STATUS_INVALID_VALUE)\
        fprintf(stderr,"'CUBLAS_STATUS_INVALID_VALUE'");\
    else if(status==CUBLAS_STATUS_ARCH_MISMATCH)\
        fprintf(stderr,"'CUBLAS_STATUS_ARCH_MISMATCH'");\
    else if(status==CUBLAS_STATUS_MAPPING_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_MAPPING_ERROR'");\
    else if(status==CUBLAS_STATUS_EXECUTION_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_EXECUTION_FAILED'");\
    else if(status==CUBLAS_STATUS_INTERNAL_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_INTERNAL_ERROR'");\
    else						 \
      fprintf(stderr,"UNKNOWN CUBLAS ERROR");\
    std::exit(EXIT_FAILURE);				     \
    }                                                        \
  }
#else 
#define CUBLAS_SAFE(expression) //expression  
#endif 


#ifdef _WITH_CUDA
#define CUDA_STREAM(cmd)({\
      cudaStream_t stream;\
      CUDA_SAFE(cudaStreamCreate(&stream));\
      cmd;						\
      CUDA_SAFE(cudaStreamSynchronize(stream));\
      CUDA_SAFE(cudaStreamDestroy(stream));\
      })
#else
#define CUDA_STREAM(cmd) CNINE_NOCUDA_ERROR
#endif


// ---- Cengine stuff ----------------------------------------------------------------------------------------


#ifdef _WITH_CENGINE
#include "Cobject.hpp"
typedef Cengine::Cobject CnineBackendObject;
#else
namespace cnine{
  class CnineBackendObject{
  };
}
#endif


#ifdef _WITH_CENGINE
#define CNINE_RSCALAR_IMPL RscalarM
#define CNINE_CSCALAR_IMPL CscalarM
#define CNINE_RTENSOR_IMPL RtensorM
#define CNINE_CTENSOR_IMPL CtensorM
#define CNINE_RTENSORARRAY_IMPL RtensorArrayM
#define CNINE_CTENSORARRAY_IMPL CtensorArrayM
#else 
#define CNINE_RSCALAR_IMPL RscalarA
#define CNINE_CSCALAR_IMPL CscalarA
#define CNINE_RTENSOR_IMPL RtensorA
#define CNINE_CTENSOR_IMPL CtensorA
#define CNINE_RTENSORARRAY_IMPL RtensorArrayA
#define CNINE_CTENSORARRAY_IMPL CtensorArrayA
#endif 


//#include "CnineSession.hpp"


#endif 
