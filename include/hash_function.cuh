#ifndef HASH_FUNCTION_CUH
#define HASH_FUNCTION_CUH

#include <stdint.h>

#include "declaration.cuh"
#include "base.h"

/////////////////////////////////////////////////////////////////////////////////////////
//Normale Hashfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
//Modulo-Funktion
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t modulo_hash(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    return hashed_key % tableSize;
};

//Multiplikative Methode
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t multiplication_hash(T value, size_t tableSize){
    double golden_value, hashed_key, hash_table_size_double;
    
    golden_value = (sqrt(5.0)-1.0)/2.0;
    hashed_key = (double) value;
    hash_table_size_double = (double) tableSize;

    hashed_key = floor(hash_table_size_double*((hashed_key*golden_value) - floor(hashed_key*golden_value)));
    
    return (size_t) hashed_key % tableSize;
};

//Universelle Hashverfahren
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t universal_hash(T value, size_t tableSize, size_t a, size_t b, size_t primeNum){
    size_t hashed_key = (size_t) value;
    return ((a*hashed_key + b)%primeNum) % tableSize;
};

//Murmer-Hash
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t murmer_hash(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;

    hashed_key ^= hashed_key >> 16;
    hashed_key *= 0x85ebca6b;
    hashed_key ^= hashed_key >> 13;
    hashed_key *= 0xc2b2ae35;
    hashed_key ^= hashed_key >> 16;
        
    return (hashed_key & (tableSize-1)) % tableSize;
};

/////////////////////////////////////////////////////////////////////////////////////////
//DyCuckoo-Hashfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t hash1(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    
    hashed_key = ~hashed_key + (hashed_key << 15);
    hashed_key = hashed_key ^ (hashed_key >> 12);
    hashed_key = hashed_key + (hashed_key << 2);
    hashed_key = hashed_key ^ (hashed_key >> 4);
    hashed_key = hashed_key * 2057;
    hashed_key = hashed_key ^ (hashed_key >> 16);

    return hashed_key % tableSize;
};

template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t hash2(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    
    hashed_key = (hashed_key + 0x7ed55d16) + (hashed_key << 12);
    hashed_key = (hashed_key ^ 0xc761c23c) ^ (hashed_key >> 19);
    hashed_key = (hashed_key + 0x165667b1) + (hashed_key << 5);
    hashed_key = (hashed_key + 0xd3a2646c) ^ (hashed_key << 9);
    hashed_key = (hashed_key+ 0xfd7046c5) + (hashed_key << 3);
    hashed_key = (hashed_key ^ 0xb55a4f09) ^ (hashed_key >> 16);
    
    return hashed_key % tableSize;
};

template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t hash3(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    return (((hashed_key ^ 59064253) + 72355969) % PRIME_uint) % tableSize;
};

template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t hash4(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;

    hashed_key = (hashed_key ^ 61) ^ (hashed_key>> 16);
    hashed_key = hashed_key + (hashed_key << 3);
    hashed_key = hashed_key ^ (hashed_key >> 4);
    hashed_key = hashed_key * 0x27d4eb2d;
    hashed_key = hashed_key ^ (hashed_key >> 15);

    return hashed_key % tableSize;
};

template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t hash5(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    hashed_key -= (hashed_key << 6);
    hashed_key ^= (hashed_key >> 17);
    hashed_key -= (hashed_key << 9);
    hashed_key ^= (hashed_key << 4);
    hashed_key -= (hashed_key << 3);
    hashed_key ^= (hashed_key << 10);
    hashed_key ^= (hashed_key >> 15);
    
    return hashed_key % tableSize;
};

/////////////////////////////////////////////////////////////////////////////////////////
//Wahl einer Hashfunktion
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t getHash(T key, size_t table_size, hash_function function){
    if (function == multiplication){
        return multiplication_hash<T>(key,table_size);
    }else if (function == universal0){
        return universal_hash<T>(key, table_size,34999950,34999960,34999969);
    }else if (function == universal1){
        return universal_hash<T>(key, table_size,15999950,15999990,15999989);
    }else if (function == universal2){
        return universal_hash<T>(key, table_size,135,140,149);
    }else if (function == murmer){
        return murmer_hash<T>(key, table_size);
    }else if (function == dycuckoo_hash1){
        return hash1<T>(key, table_size);
    }else if (function == dycuckoo_hash2){
        return hash2<T>(key, table_size);
    }else if (function ==dycuckoo_hash3){
        return hash3<T>(key, table_size);
    }else if (function ==dycuckoo_hash4){
        return hash4<T>(key, table_size);
    }else if (function ==dycuckoo_hash5){
        return hash5<T>(key, table_size);
    }else{
        return modulo_hash<T>(key,table_size);
    }
};

//Vertausche zwischen zwei Schlüsseln
template <typename T>
HOSTQUALIFIER T swapHash(T currentKey, T reference, T key){
    if (currentKey==reference){
        currentKey = key;
        return currentKey;
    }else{
        return currentKey;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
//Sondierungsfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
//Berechne einen Sondierungswert eines Schlüssels durch eine andere Hashfunktion
template <typename T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t getHashProbe(T key, size_t i, size_t table_size, hash_function function){
    return i*getHash(key,table_size,function);
};

//Quadratische Sondierungsfunktion
HOSTDEVICEQUALIFIER INLINEQUALIFIER size_t getProbe2(size_t i){
    size_t j = (size_t) pow(ceil((double)i/2),2.0);
    size_t k = (size_t) pow(-1.0,(double)i);
    return (j * k);
};

#endif