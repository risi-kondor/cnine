//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RtensorUtils_cu
#define _RtensorUtils_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "CUDAhelpers.hpp"


// ---- Rtensor Get/set ---------------------------------------------------------------------------------------


//__global__ float Rtensor_get_kernel(const float* arr){
//  return *arr;
//}

//__global__ void Rtensor_set_kernel(float* arr, const float v){
//  *arr=v;
//}

__global__ void Rtensor_inc_kernel(float* arr, const float v){
  *arr+=v;
}


// ---- Rtensor1 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_t(float* rarr, const float* arr, 
  const int s0, const int rs0){
  rarr[threadIdx.x*rs0]=arr[threadIdx.x*s0];
}


// ---- Rtensor1 add -----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_t(float* rarr, const float* arr, 
  const int s0, const int rs0){
  rarr[threadIdx.x*rs0]+=arr[threadIdx.x*s0];
}


// ---- Rtensor2 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_tt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]=arr[threadIdx.x*s0+threadIdx.y*s1];
}

__global__ void Rtensor_copy_kernel_bt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]=arr[blockIdx.x*s0+threadIdx.x*s1];
}

__global__ void Rtensor_copy_kernel_bb(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1]=arr[blockIdx.x*s0+blockIdx.y*s1];
}


// ---- Rtensor2 add -----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_tt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=arr[threadIdx.x*s0+threadIdx.y*s1];
}

__global__ void Rtensor_add_kernel_bt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=arr[blockIdx.x*s0+threadIdx.x*s1];
}

__global__ void Rtensor_add_kernel_bb(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1]+=arr[blockIdx.x*s0+blockIdx.y*s1];
}


// ---- Rtensor3 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2];
}

__global__ void Rtensor_copy_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2];
}

__global__ void Rtensor_copy_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2];
}

__global__ void Rtensor_copy_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2];
}


// ---- Rtensor3 add ----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2];
}

__global__ void Rtensor_add_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2];
}

__global__ void Rtensor_add_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2];
}

__global__ void Rtensor_add_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2];
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{

  
  float Rtensor_get_cu(const float* p){
    float r=0;
    CUDA_SAFE(cudaMemcpy(&r,p,sizeof(float),cudaMemcpyDeviceToHost));
    return r;
  }

  void Rtensor_set_cu(float* p, const float& v){
    //Rtensor_set_kernel<<<1,1>>>(p,v);
    CUDA_SAFE(cudaMemcpy(p,&v,sizeof(float),cudaMemcpyHostToDevice));
  }

  void Rtensor_inc_cu(float* p, const float v){
    Rtensor_inc_kernel<<<1,1>>>(p,v);
  }


  void Rtensor_copy_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream){
    if(x.n0<1024){
      Rtensor_copy_kernel_t<<<0,x.n0,0,stream>>>(r.arr,x.arr,x.s0,r.s0);
      return;
    }
    Rtensor_copy_kernel_bt<<<x.n0/1024,1024,0,stream>>>(r.arr,x.arr,1024*x.s0,x.s0,1024*r.s0,r.s0);},
    Rtensor_copy_kernel_t<<<0,x.n0%1024,0,stream>>>(r.arr+(x.n0-x.n0%1024)*r.s0,x.arr+(x.n0-x.n0%1024),x.s0,r.s0);
  }

  void Rtensor_copy_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_tt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_bt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_bb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);});
  }

  void Rtensor_copy_cu(const Rtensor3_view& r, const Rtensor3_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_ttt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_btt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_bbt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_bbb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);});
  }


  void Rtensor_add_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream){
    if(x.n0<1024){
      Rtensor_add_kernel_t<<<0,x.n0,0,stream>>>(r.arr,x.arr,x.s0,r.s0);
      return;
    }
    Rtensor_add_kernel_bt<<<x.n0/1024,1024,0,stream>>>(r.arr,x.arr,1024*x.s0,x.s0,1024*r.s0,r.s0);},
    Rtensor_add_kernel_t<<<0,x.n0%1024,0,stream>>>(r.arr+(x.n0-x.n0%1024)*r.s0,x.arr+(x.n0-x.n0%1024),x.s0,r.s0);
  }

  void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_tt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);});
  }

  void Rtensor_add_cu(const Rtensor3_view& r, const Rtensor3_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_ttt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_btt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);});
  }

}



__global__ void batched_add_kernel_0(float* rarr, const float* arr, const int sb, const int s0){
  rarr[blockIdx.x*sb+threadIdx.x*s0]+=arr[blockIdx.x*sb+threadIdx.x*s0];  
}


__global__ void batched_add_kernel_1(float* rarr, const float* arr, const int sb, const int s0, const int s1){
  rarr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1]+=arr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1];  
}



namespace cnine{


  void batched_add_cu(float* rarr, const float* arr, const int b, const int sb, const int n, const int s, const cudaStream_t& stream){

    if(n<=1024){
      batched_add_kernel_0<<<b,n,0,stream>>>(rarr,arr,sb,s);
    }else{
      dim3 blocks(b,n/1024);
      batched_add_kernel_1<<<blocks,1024,0,stream>>>(rarr,arr,sb,s*1024,s);
      if(n%1024>0) batched_add_kernel_0<<<b,n%1024,0,stream>>>(rarr+(n-n%1024)*s,arr+(n-n%1024)*s,sb,s);
    }
 
  }


}

#endif 

  /*
  void Rtensor_copy_cu(const Rtensor2_view& r, const Rtensor2_view* x, const cudaStream_t& stream){

    if(x.n0*x.n1<=1024){
      dim3 threads(x.n0,x.n1);
      Rtensor_copy_kernel_tt<<<1,threads,0,stream>>>(r.arr,xarr,x.s0,x.s1,r.s0,r.s1);
      return;
    }

    if(x.n1<=1024){
      Rtensor_copy_kernel_bt<<<x.n0,x.n1,0,stream>>>(r.arr,xarr,x.s0,x.s1,r.s0,r.s1);
      return;
    }

    if(x.n0<=1024){
      Rtensor_copy_kernel_bt<<<x.n1,x.n0,0,stream>>>(r.arr,xarr,x.s1,x.s0,r.s1,r.s0);
      return;
    }

    CNINE_UNIMPL();

  }
  */
  /*
  void Rtensor_copy_cu(const Rtensor3_view& r, const Rtensor3_view* x, const cudaStream_t& stream){

    if(x.n0*x.n1*x.n2<=1024){
      dim3 threads(x.n0,x.n1,x.n2);
      Rtensor_copy_kernel_ttt<<<1,threads,0,stream>>>(r.arr,xarr,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(x.n1*x.n2<=1024){
      dim3 threads(x.n1,x.n2);
      Rtensor_copy_kernel_btt<<<x.n0,threads,0,stream>>>(r.arr,xarr,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(x.n0*x.n2<=1024){
      dim3 threads(x.n0,x.n2);
      Rtensor_copy_kernel_btt<<<x.n1,threads,0,stream>>>(r.arr,xarr,x.s1,x.s0,x.s2,r.s1,r.s0,r.s2);
      return;
    }

    if(x.n0*x.n1<=1024){
      dim3 threads(x.n0,x.n1);
      Rtensor_copy_kernel_btt<<<x.n2,threads,0,stream>>>(r.arr,xarr,x.s2,x.s0,x.s1,r.s2,r.s0,r.s1);
      return;
    }

    if(x.n2<=1024){
      dim3 blocks(x.n0,x.n1);
      Rtensor_copy_kernel_bbt<<<x.n1,x.n0,0,stream>>>(r.arr,xarr,x.s1,x.s0,r.s1,r.s0);
      return;
    }

    CNINE_UNIMPL();

  }
  */
  /*
  void Rtensor_copy_cu(const Rtensor3_view& r, const Rtensor3_view* x, const cudaStream_t& stream){

    if(x.n0*x.n1*x.n2<=1024){
      dim3 threads(x.n0,x.n1,x.n2);
      Rtensor_copy_kernel_ttt<<<1,threads,0,stream>>>(r.arr,xarr,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(x.n1*x.n2<=1024){
      dim3 threads(x.n1,x.n2);
      Rtensor_copy_kernel_btt<<<x.n0,threads,0,stream>>>(r.arr,xarr,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(x.n0*x.n2<=1024){
      dim3 threads(x.n0,x.n2);
      Rtensor_copy_kernel_btt<<<x.n1,threads,0,stream>>>(r.arr,xarr,x.s1,x.s0,x.s2,r.s1,r.s0,r.s2);
      return;
    }

    if(x.n0*x.n1<=1024){
      dim3 threads(x.n0,x.n1);
      Rtensor_copy_kernel_btt<<<x.n2,threads,0,stream>>>(r.arr,xarr,x.s2,x.s0,x.s1,r.s2,r.s0,r.s1);
      return;
    }

    if(x.n2<=1024){
      dim3 blocks(x.n0,x.n1);
      Rtensor_copy_kernel_bbt<<<x.n1,x.n0,0,stream>>>(r.arr,xarr,x.s1,x.s0,r.s1,r.s0);
      return;
    }

    CNINE_UNIMPL();

  }
  */
  /*
  void Rtensor2_view_add_cu(float* rarr, const float* arr, const int b, const int sb, const int n, const int s, const cudaStream_t& stream){

    if(n<=1024){
      batched_add_kernel_0<<<b,n,0,stream>>>(rarr,arr,sb,s);
      return;
    }

    dim3 blocks(b,n/1024);
    int offs=(n/1024)*1024;
    batched_add_kernel_1<<<blocks,1024,0,stream>>>(rarr,arr,sb,s*1024,s);
    if(n%1024>0) batched_add_kernel_0<<<b,n%1024,0,stream>>>(rarr+(n-n%1024)*s,arr+(n-n%1024)*s,sb,s);

  }
  */
