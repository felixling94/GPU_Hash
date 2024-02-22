#ifndef DECLARATION_CUH
#define DECLARATION_CUH

#ifdef __CUDACC__
#define HOSTDEVICEQUALIFIER  __device__  __host__ 
#else
#define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
#define GLOBALQUALIFIER  __global__
#else
#define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
#define DEVICEQUALIFIER  __device__
#else
#define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
#define HOSTQUALIFIER  __host__
#else
#define HOSTQUALIFIER
#endif

#ifdef __CUDACC__
#define INLINEQUALIFIER  __forceinline__
#else
#define INLINEQUALIFIER inline
#endif

#define PRIME_uint 294967291u

#endif