#ifndef DYCUCKOO_HASHFUNKTIONEN_CUH
#define DYCUCKOO_HASHFUNKTIONEN_CUH

#include <stdint.h>

#include <../include/deklaration.cuh>

#define PRIME_uint 294967291u

namespace DyCuckoo_Funktionen_Device{
    //Berechne den Hashwert eines Schlüssels durch verschiedene Hashfunktione aus DyCuckoo
    template <typename T>
    DEVICEQUALIFIER INLINEQUALIFIER size_t hash1(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        schluessel_hash = ~schluessel_hash + (schluessel_hash << 15);
        schluessel_hash = schluessel_hash ^ (schluessel_hash >> 12);
        schluessel_hash = schluessel_hash + (schluessel_hash << 2);
        schluessel_hash = schluessel_hash ^ (schluessel_hash >> 4);
        schluessel_hash = schluessel_hash * 2057;
        schluessel_hash = schluessel_hash ^ (schluessel_hash >> 16);

        return schluessel_hash;
    };

    template <typename T>
    DEVICEQUALIFIER INLINEQUALIFIER size_t hash2(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        schluessel_hash = (schluessel_hash + 0x7ed55d16) + (schluessel_hash << 12);
        schluessel_hash = (schluessel_hash ^ 0xc761c23c) ^ (schluessel_hash >> 19);
        schluessel_hash = (schluessel_hash + 0x165667b1) + (schluessel_hash << 5);
        schluessel_hash = (schluessel_hash + 0xd3a2646c) ^ (schluessel_hash << 9);
        schluessel_hash = (schluessel_hash+ 0xfd7046c5) + (schluessel_hash << 3);
        schluessel_hash = (schluessel_hash ^ 0xb55a4f09) ^ (schluessel_hash >> 16);
    
        return schluessel_hash;
    };

    template <typename T>
    DEVICEQUALIFIER INLINEQUALIFIER size_t hash3(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        return ((schluessel_hash ^ 59064253) + 72355969) % PRIME_uint;
    };

    template <typename T>
    DEVICEQUALIFIER INLINEQUALIFIER size_t hash4(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        schluessel_hash = (schluessel_hash ^ 61) ^ (schluessel_hash>> 16);
        schluessel_hash = schluessel_hash + (schluessel_hash << 3);
        schluessel_hash = schluessel_hash ^ (schluessel_hash >> 4);
        schluessel_hash = schluessel_hash * 0x27d4eb2d;
        schluessel_hash = schluessel_hash ^ (schluessel_hash >> 15);

        return schluessel_hash;
    };

    template <typename T>
    DEVICEQUALIFIER INLINEQUALIFIER size_t hash5(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        schluessel_hash -= (schluessel_hash << 6);
        schluessel_hash ^= (schluessel_hash >> 17);
        schluessel_hash -= (schluessel_hash << 9);
        schluessel_hash ^= (schluessel_hash << 4);
        schluessel_hash -= (schluessel_hash << 3);
        schluessel_hash ^= (schluessel_hash << 10);
        schluessel_hash ^= (schluessel_hash >> 15);
    
        return schluessel_hash;
    };
};

#endif