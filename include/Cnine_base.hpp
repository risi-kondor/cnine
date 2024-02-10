/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _Cnine_base
#define _Cnine_base

#include <assert.h>

#include <complex>
#include <iostream>
#include <map>
#include <unordered_map>
#include <random>
#include <functional> 
#include <thread>
#include <mutex>
#include <array>
#include <set>
#include <list>
#include <memory>
#include <algorithm>

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 


using namespace std; 

#define CNINE_ERROR(message) \
  throw std::runtime_error("Cnine error in "+string(__PRETTY_FUNCTION__)+" : "+message+".");

#define CNINE_ASSRT(condition) \
  if(!(condition)) throw std::runtime_error("Cnine error in "+string(__PRETTY_FUNCTION__)+" : failed assertion "+#condition+".");

#define CNINE_ASSERT(condition, message) \
  if(!(condition)) throw std::runtime_error("Cnine error in "+string(__PRETTY_FUNCTION__)+": "+message+".");

//if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }

#define CNINE_UNIMPL() printf("Cnine error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);


// ---- Copy, assign and convert warnings --------------------------------------------------------------------


#ifdef CNINE_COPY_WARNINGS
#define CNINE_COPY_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" copied."<<endl;
#else 
#define CNINE_COPY_WARNING()
#endif 

#ifdef CNINE_ASSIGN_WARNINGS
#define CNINE_ASSIGN_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" assigned."<<endl;
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

#ifdef CNINE_CONVERT_WARNINGS
#define CNINE_CONVERT_WARNING() cout<<"\e[1mcnine:\e[0m conversion in "<<string(__PRETTY_FUNCTION__)<<" ."<<endl;
#else 
#define CNINE_CONVERT_WARNING()
#endif 

#ifdef CNINE_ATEN_CONVERT_WARNINGS
#define CNINE_CONVERT_FROM_ATEN_WARNING() cout<<"\e[1mcnine:\e[0m ATen tensor converted to "<<classname()<<"."<<endl;
#define CNINE_CONVERT_TO_ATEN_WARNING() cout<<"\e[1mcnine:\e[0m "<<classname()<<" converted to ATen tensor."<<endl;
#else 
#define CNINE_CONVERT_FROM_ATEN_WARNING()
#define CNINE_CONVERT_TO_ATEN_WARNING()
#endif 


// ---- Range checking ---------------------------------------------------------------------------------------


#ifdef CNINE_RANGE_CHECKING
#define CNINE_CHECK_RANGE(expr) expr
#define CNINE_CHECK_SIZE(expr) expr
#define CNINE_IN_RANGE(ix,tsize) if(ix>=tsize) throw std::out_of_range("Cnine error in "+string(__PRETTY_FUNCTION__)+": index "+to_string(ix)+" out of range (0,"+to_string(tsize-1)+").");
#define CNINE_DIMS(d) if(dims.size()!=d) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": number of dimensions in "+dims.str()+" is not "+to_string(d)+".");
#define CNINE_DIMS_VALID(dims) if(!dims.valid()) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": invalid dimensions"+dims.str()+".");
#define CNINE_DIMS_SAME(x) if(x.dims!=dims) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": dimension mismatch between "+dims.str()+" and "+x.dims.str()+".");
#define CNINE_DIMS_EQ(a,b) if(a!=b) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": dimension mismatch between "+a.str()+" and "+b.str()+".");
#define CNINE_DIMS_EQ_TOTAL(a,b) if(a.total()!=b.total()) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": mismatch between total size of "+a.str()+" and "+b.str()+".");
#define CNINE_NDIMS_IS_1(a) if(a.dims.size()!=1) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": tensor is not a vector."); 
#define CNINE_NDIMS_IS_2(a) if(a.dims.size()!=2) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": tensor is not a matrix."); 
#define CNINE_NDIMS_IS(n) if(dims.size()!=n) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": tensor is of order "+to_string(dims.size())+"."); 
#define CNINE_NDIMS_LEAST(n) if(dims.size()<n) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": tensor is expected to be of order "+to_string(n)+" but is of order "+to_string(dims.size())+"."); 
#define CNINE_NDIMS_LEASTX(x,n) if(x.dims.size()<n) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": tensor is expected to be of order "+to_string(n)+" but is of order "+to_string(x.dims.size())+"."); 
//#define CNINE_CHECK_DIM(d,i) if(dims[d]<=i) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": "+to_string(i)+" is out of bounds for dimension "+to_string(d)+" in "+dims.str()+"."); 
#define CNINE_NTENS_SAME(x) if(x.tensors.size()!=tensors.size()) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": mismatch in number of tensors "+to_string(x.tensors.size())+" vs "+to_string(tensors.size())+".");
#else
#define CNINE_CHECK_RANGE(expr)
#define CNINE_CHECK_SIZE(expr)
#define CNINE_DIMS(d)
#define CNINE_IN_RANGE(ix,tsize)
#define CNINE_DIMS_VALID(dims)
#define CNINE_DIMS_SAME(x)
#define CNINE_DIMS_EQ(a,b)
#define CNINE_DIMS_EQ_TOTAL(a,b)
#define CNINE_NDIMS_IS_1(a)
#define CNINE_NDIMS_IS_2(a)
#define CNINE_NDIMS_IS(n)
#define CNINE_NDIMS_LEAST(n)
#define CNINE_NDIMS_LEASTX(x,n)
//#define CNINE_CHECK_DIM(d,i)
#define CNINE_NTENS_SAME(x)
#endif



// ---- Device checking ---------------------------------------------------------------------------------------


#ifdef CNINE_DEVICE_CHECKING
#define CNINE_CHECK_DEV(expr) expr
#define CNINE_DEVICE_VALID(dev) if(dev<0 || dev>1) throw std::invalid_argument("Cnine error in "+string(__PRETTY_FUNCTION__)+": device must be 0 or 1.");
#define CNINE_DEVICE_SAME(x) if(x.dev!=dev) throw std::out_of_range("Cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");
#define CNINE_DEVICE_SAMEB(x) if(x.get_dev()!=get_dev()) throw std::out_of_range("Cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");
#define CNINE_DEVICE_EQ(x,y) if(x.dev!=y.dev) throw std::out_of_range("Cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");
#else
#define CNINE_CHECK_DEV(expr)
#define CNINE_DEVICE_VALID(dev) 
#define CNINE_DEVICE_SAME(x) 
#define CNINE_DEVICE_SAMEB(x) 
#define CNINE_DEVICE_EQ(x,y) 
#endif

#define CNINE_CHECK_DEV2(x,y) if(x.dev!=y.dev) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");
#define CNINE_CHECK_DEV3(x,y,z) if(x.dev!=y.dev || x.dev!=z.dev) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": device mismatch.");


// ---- Templates ---------------------------------------------------------------------------------------------


#define IF_INT template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,int>::value, U>::type>
#define IF_FLOAT template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,float>::value, U>::type>
#define IF_DOUBLE template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,double>::value, U>::type>
#define IF_CFLOAT template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,complex<float> >::value, U>::type>



// ---- other -------------------------------------------------------------------------------------------------


#define CNINE_NOCUDA_ERROR cout<<"Error: Cnine was compiled without GPU support."<<endl;
#define CNINE_CPUONLY() if(dev!=0) {printf("Cnine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}
#define CNINE_CPUONLY1(x) if(x.dev!=0) {printf("Cnine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}

#define COUT(cmd) {cnine::CoutLock lk; cout<<cmd<<endl;}
#define CNINE_COUT(cmd) {CoutLock lk; cout<<cmd<<endl;}

#define CNINE_CHECK_DIMS(a,b,fn) if(a!=b) {{CoutLock lk; cerr<<"cnine error in function "<<fn<<": dimension mismatch."<<endl;} exit(1);}
#define CNINE_CHECK_DIMS2(a,b,c,fn) if(a!=b||a!=c) {{CoutLock lk; cerr<<"cnine error in function "<<fn<<": dimension mismatch."<<endl;} exit(1);}

#define CNINE_CHECK_BATCH2(x,y) if(x.n0!=y.n0) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": batch dimension mismatch.");
#define CNINE_CHECK_BATCH3(x,y,z) if(x.n0!=y.n0 || x.n0!=z.n0) throw std::out_of_range("cnine error in "+std::string(__PRETTY_FUNCTION__)+": batch dimension mismatch.");

#define BLOB_DEBUG(str) {CoutLock lk; cerr<<str<<endl;}
//#define BLOB_DEBUG(str)


// ---- tracing ----------------------------------------------------------------------------------------------


#ifdef CNINE_FUNCTION_TRACING
#define FNTRACE() tracer fntracer(__PRETTY_FUNCTION__);
#else 
#define FNTRACE()
#endif 


namespace cnine{


  template<class T> struct is_complex : std::false_type {};
  template<class T> struct is_complex<std::complex<T>> : std::true_type {};
  
  //template<typename S, typename=typename std::enable_if<is_complex<S>::value, S>::type>
  //struct is_complex: public true_type{};

  //template<typename S, typename=typename std::enable_if<!is_complex<S>::value, S>::type>
  //struct is_complex: public false_type{};


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
  struct fill_reserve: public fill_pattern{fill_reserve(){}};
  struct fill_tensor: public fill_pattern{fill_tensor(){}};
  struct fill_stack: public fill_pattern{fill_stack(){}};
  struct fill_cat: public fill_pattern{fill_cat(){}};
  struct fill_cgaussian: public fill_pattern{fill_cgaussian(){}};
  struct fill_random_unitary: public fill_pattern{fill_random_unitary(){}};

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

  template<typename TYPE> 
  struct fill_constant: public fill_pattern{
    TYPE v=0;
    fill_constant(){}
    fill_constant(const TYPE _v):v(_v){}
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
    static const fill_random_unitary random_unitary;
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


  // --- Devices ---------------------------------------------------------------------------------------------


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
    static device CPU(0);
    static device GPU0(1);
  } 

  class DeviceSelector{
  public:
    int dev=0;
    int max_mem=1024;
  };


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
    lock_guard<mutex> lock;
    static std::mutex mx;
  };


  // ---- Helper classes -------------------------------------------------------------------------------------


  struct size_spec: public fill_pattern{
    const int n;
    size_spec(const int _n): n(_n){}
  };


  template<typename TYPE>
  class triple{
  public:
    TYPE first;
    TYPE second;
    TYPE third;
  public:
    triple(const TYPE& _first, const TYPE& _second, const TYPE& _third):
      first(_first), second(_second), third(_third){}
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

  template<typename TYPE>
  std::shared_ptr<TYPE> to_share(TYPE* x){
    return std::shared_ptr<TYPE>(x);
  }

  template<typename TYPE>
  class selector: public std::vector<TYPE>{
    TYPE operator()(const int i) const {return (*this)[i];}
  };

  class Printable{
  public:
    virtual string str(const string ident="") const=0;
    friend ostream& operator<<(ostream& stream, const Printable& x){
      stream<<x.str(); return stream;}
  };


  template<typename TYPE>
  class _batched{
  public:
    const TYPE& x;
    _batched(const TYPE& _x):x(_x){}
    operator TYPE(){return x;}
  };

  //template<typename TYPE>
  //_batched<TYPE> batch(const TYPE& x){return x;}
    

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

  //template<typename TYPE>
  //inline TYPE max(const initializer_list<pair<TYPE,TYPE> >& list)

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

  #define PRINTL(x) printl(#x,x);

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
      cudaStream_t stream=NULL;			\
      CUDA_SAFE(cudaStreamCreate(&stream));		\
      cmd;						\
      CUDA_SAFE(cudaStreamSynchronize(stream));\
      CUDA_SAFE(cudaStreamDestroy(stream));\
      })
#else
#define CUDA_STREAM(cmd) CNINE_NOCUDA_ERROR
#endif
      //      cudaStream_t stream=NULL;			\



#ifdef _WITH_CUDA
class cu_stream{
public:

  cudaStream_t stm;				\

  cu_stream(){
    CUDA_SAFE(cudaStreamCreate(&stm));
  }

  ~cu_stream(){
    CUDA_SAFE(cudaStreamSynchronize(stm));
    CUDA_SAFE(cudaStreamDestroy(stm));
  }

  operator cudaStream_t() const{
    return stm;
  }

};
#endif 

#define CPUCODE(cmd) if(dev==0){cmd;}

#ifdef _WITH_CUDA
#define GPUCODE(cmd) if(dev==1){cmd;}
#else
#define GPUCODE(cmd) 
#endif

namespace cnine{

  template<typename TYPE>
  class ArrayOnDevice{
  public:

    TYPE* arr;

    ArrayOnDevice(const int _n){
      CUDA_SAFE(cudaMalloc((void **)&arr, std::max(_n*sizeof(TYPE),(unsigned long) 1)));
    }

    ~ArrayOnDevice(){
      CUDA_SAFE(cudaFree(arr));
    }

    operator TYPE*() const{
      return arr;
    }
  };

}

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
#define CNINE_CTENSOR_IMPL CtensorB
#define CNINE_RTENSORARRAY_IMPL RtensorArrayA
#define CNINE_CTENSORARRAY_IMPL CtensorArrayB
#endif 


//#include "CnineSession.hpp"


#endif 
