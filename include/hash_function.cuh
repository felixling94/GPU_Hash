#ifndef HASH_FUNCTION_CUH
#define HASH_FUNCTION_CUH

#include "base.h"

/////////////////////////////////////////////////////////////////////////////////////////
//Normale Hashfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
//Divisions-Rest-Methode
template <typename T>
__device__  __host__ size_t modulo_hash(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    return hashed_key % tableSize;
};

//Multiplikative Methode
template <typename T>
__device__  __host__ size_t multiplication_hash(T value, size_t tableSize){
    double golden_value, hashed_key, hash_table_size_double;
    
    golden_value = (sqrt(5.0)-1.0)/2.0;
    hashed_key = (double) value;
    hash_table_size_double = (double) tableSize;

    hashed_key = floor(hash_table_size_double*((hashed_key*golden_value) - floor(hashed_key*golden_value)));
    
    return (size_t) hashed_key % tableSize;
};

//Universelle Hashverfahren
template <typename T>
__device__  __host__ size_t universal_hash(T value, size_t tableSize, size_t a, size_t b, size_t primeNum){
    size_t hashed_key = (size_t) value;
    return ((a*hashed_key + b)%primeNum) % tableSize;
};

//Murmer-Hashfunktion
template <typename T>
__device__  __host__ size_t murmer_hash(T value, size_t tableSize){
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
//DyCuckoo-1
template <typename T>
__device__  __host__ size_t hash1(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    
    hashed_key = ~hashed_key + (hashed_key << 15);
    hashed_key = hashed_key ^ (hashed_key >> 12);
    hashed_key = hashed_key + (hashed_key << 2);
    hashed_key = hashed_key ^ (hashed_key >> 4);
    hashed_key = hashed_key * 2057;
    hashed_key = hashed_key ^ (hashed_key >> 16);

    return hashed_key % tableSize;
};

//DyCuckoo-2
template <typename T>
__device__  __host__ size_t hash2(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    
    hashed_key = (hashed_key + 0x7ed55d16) + (hashed_key << 12);
    hashed_key = (hashed_key ^ 0xc761c23c) ^ (hashed_key >> 19);
    hashed_key = (hashed_key + 0x165667b1) + (hashed_key << 5);
    hashed_key = (hashed_key + 0xd3a2646c) ^ (hashed_key << 9);
    hashed_key = (hashed_key+ 0xfd7046c5) + (hashed_key << 3);
    hashed_key = (hashed_key ^ 0xb55a4f09) ^ (hashed_key >> 16);
    
    return hashed_key % tableSize;
};

//DyCuckoo-3
template <typename T>
__device__  __host__ size_t hash3(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;
    return (((hashed_key ^ 59064253) + 72355969) % 294967291u) % tableSize;
};

//DyCuckoo-4
template <typename T>
__device__  __host__ size_t hash4(T value, size_t tableSize){
    size_t hashed_key = (size_t) value;

    hashed_key = (hashed_key ^ 61) ^ (hashed_key>> 16);
    hashed_key = hashed_key + (hashed_key << 3);
    hashed_key = hashed_key ^ (hashed_key >> 4);
    hashed_key = hashed_key * 0x27d4eb2d;
    hashed_key = hashed_key ^ (hashed_key >> 15);

    return hashed_key % tableSize;
};

//DyCuckoo-5
template <typename T>
__device__  __host__ size_t hash5(T value, size_t tableSize){
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
__device__  __host__ size_t getHash(T value, size_t table_size, hash_function function){
    if (function == multiplication){
        return multiplication_hash<T>(value,table_size);
    }else if (function == universal0 || function == universal3){
        return universal_hash<T>(value, table_size, 20019, 20025, 20029);
    }else if (function == universal1){
        return universal_hash<T>(value, table_size, 10023, 10037, 10039);
    }else if (function == universal2){
        return universal_hash<T>(value, table_size, 5029, 5038, 5039);
    }else if (function == murmer){
        return murmer_hash<T>(value, table_size);
    }else if (function == dycuckoo_hash1){
        return hash1<T>(value, table_size);
    }else if (function == dycuckoo_hash2){
        return hash2<T>(value, table_size);
    }else if (function ==dycuckoo_hash3){
        return hash3<T>(value, table_size);
    }else if (function ==dycuckoo_hash4){
        return hash4<T>(value, table_size);
    }else if (function ==dycuckoo_hash5){
        return hash5<T>(value, table_size);
    }else{
        return modulo_hash<T>(value,table_size);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
//Vertausche eine Zelle mit der anderen in einer Hashtabelle bei Cuckoo-Hashverfahren
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
__device__ void swapCells(T1 key, T2 value, int i, cell<T1,T2> * hash_table){
    T1 temp_key = hash_table[i].key;
    T2 temp_value = hash_table[i].value;
    
    hash_table[i].key = key;
    hash_table[i].value = value;
    
    key = temp_key;
    value = temp_value;
};

/* Vertausche eine Zelle mit der anderen in einer Hashtabelle, 
    wobei die Funktionalität der von atomicCAS auf der GPU gleich ist 
*/
template <typename T>
 __host__ T swapHash(T currentValue, T reference, T value){
    if (currentValue==reference){
        currentValue = value;
        return currentValue;
    }else{
        return currentValue;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
//Sondierungsfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
/* Sondierungsfunktion bei doppelten Hashverfahren:
    - einen Sondierungswert eines Schlüssels durch eine andere Hashfunktion ermitteln
*/
template <typename T>
__device__  __host__ size_t getHashProbe(T value, size_t i, size_t table_size, hash_function function){
    return i*getHash<T>(value,table_size,function);
};

/* Quadratische Sondierungsfunktion
    - einen Sondierungswert eines Schlüssels durch quadratische Erhöhung vom
      Hashwert des Schlüssels ermitteln
*/
__device__  __host__ __forceinline__ int getProbe2(size_t i){
    int j = pow(ceil((double)i/2),2.0);
    int k = pow(-1.0,(double)i);
    return (j * k);
};

#endif