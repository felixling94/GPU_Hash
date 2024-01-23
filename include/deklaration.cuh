#ifndef DEKLARATION_CUH
#define DEKLARATION_CUH

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

#endif