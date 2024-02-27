#ifndef HASH_METHOD_CUH
#define HASH_METHOD_CUH

#include <stdint.h>

#include <../include/base.h>
#include <../include/declaration.cuh>
#include <../include/hash_function.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
DEVICEQUALIFIER void insert_normal(T1 key, T2 value, size_t i, cell<T1,T2>* HashTable){
    T1 prev = atomicCAS(&HashTable[i].key, BLANK, key);
    
    if (prev == BLANK || prev == key){
        HashTable[i].key = key;
        HashTable[i].value = value;
        __syncthreads();
        return;
    }
};

//Lineare Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void insert_linear(T1 key, T2 value, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j;
    size_t max_hash_table_size;
    T1 prev;

    j = 0;
    max_hash_table_size = (size_t)((100+LOOP_PERCENTAGE)/100*HashTableSize);

    while(j<max_hash_table_size){
        i = (i + j) % HashTableSize;
        prev = atomicCAS(&HashTable[i].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            HashTable[i].key = key;
            HashTable[i].value = value;
            __syncthreads();
            break;
        }
        ++j;
    }
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void insert_quadratic(T1 key, T2 value, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j;
    size_t max_hash_table_size;
    T1 prev;

    j = 0;
    max_hash_table_size = (size_t)((100+LOOP_PERCENTAGE)/100*HashTableSize);

    while(j<max_hash_table_size){
        i = ((size_t) ((int) i + getProbe2(j))) % HashTableSize;
        prev = atomicCAS(&HashTable[i].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            HashTable[i].key = key;
            HashTable[i].value = value;
            __syncthreads();
            break;
        }
        ++j;
    }
};

//Doppelte Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void insert_double(T1 key, T2 value, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize, hash_function function){
    size_t j;
    size_t max_hash_table_size;
    T1 prev;

    j = 0;
    max_hash_table_size = (size_t)((100+LOOP_PERCENTAGE)/100*HashTableSize);

    while(j<max_hash_table_size){
        i = (i + getHashProbe(key,j,HashTableSize,function)) % HashTableSize;
        prev = atomicCAS(&HashTable[i].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            HashTable[i].key = key;
            HashTable[i].value = value;
            __syncthreads();
            break;
        }
        ++j;
    }
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void insert_cuckoo(T1 key, T2 value, size_t i, size_t j, cell<T1,T2>* HashTable1, cell<T1,T2>* HashTable2, size_t HashTableSize, hash_function function){
    size_t k;
    size_t max_hash_table_size;
    T1 prev1, prev2, temp_key;
    T2 temp_value;

    k = 1;
    max_hash_table_size = (size_t)((100+LOOP_PERCENTAGE)/100*HashTableSize);

    prev1 = atomicCAS(&HashTable1[i].key, BLANK, key);
    
    if (prev1 == BLANK || prev1 == key){
    //if (HashTable1[i].key == BLANK){
    //if (HashTable1[i].key == BLANK || HashTable1[i].key == key){
        HashTable1[i].key = key;
        HashTable1[i].value = value;
        __syncthreads();
        return;
    }

    prev2 = atomicCAS(&HashTable2[i].key, BLANK, key);

    if (prev2 == BLANK || prev2 == key){
    //if (HashTable2[j].key == BLANK){
    //if (HashTable2[j].key == BLANK || HashTable2[j].key == key){
        HashTable2[j].key = key;
        HashTable2[j].value = value;
        __syncthreads();
        return;
    }
    
    while (k<max_hash_table_size){
        i = (i + k) % (2*HashTableSize);
        j = (j + k) % (2*HashTableSize);
        
        //Vertausche zwei Schlüssel innerhalb der 1. Hashtabelle
        swapCells<T1,T2>(key,value,i,HashTable1);

        prev1 = atomicCAS(&HashTable1[i].key, BLANK, key);
        
        if (prev1 == BLANK || prev1 == key){
        //if (HashTable1[i].key == BLANK){
        //if (HashTable1[i].key == BLANK || HashTable1[i].key == key){
            HashTable1[i].key = key;
            HashTable1[i].value = value;
            __syncthreads();
            break;
        }

        //Vertausche zwei Schlüssel innerhalb der 2. Hashtabelle
        swapCells<T1,T2>(key,value,j,HashTable2);

        prev2 = atomicCAS(&HashTable2[j].key, BLANK, key);

        if (prev2 == BLANK || prev2 == key){
        //if (HashTable2[j].key == BLANK){
        //if (HashTable2[j].key == BLANK || HashTable2[j].key == key){
            HashTable2[j].key = key;
            HashTable2[j].value = value;
            __syncthreads();
            break;
        }

        ++k;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
//Suchen nach einer Liste von Schlüsseln in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
DEVICEQUALIFIER T1 search_normal(T1 key, size_t i, cell<T1,T2>* HashTable){
    if (HashTable[i].key == key){
        return BLANK;
    }else{
        return key;
    }
};

//Lineare Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER T1 search_linear(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j;
    j = 0;

    while(j<HashTableSize){
        i = (i + j) % HashTableSize;
        if (HashTable[i].key == key) return BLANK;
        ++j;
    }
    return key;
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER T1 search_quadratic(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j;
    j = 0;

    while((j/2)<HashTableSize){
        i = ((size_t) ((int) i + getProbe2(j))) % HashTableSize;
        if (HashTable[i].key == key) return BLANK;
        ++j;
    }
    return key;
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER T1 search_double(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize, hash_function function){
    size_t j;
    j = 0;
    
    while(j<HashTableSize){
        i = (i + getHashProbe(key,j,HashTableSize,function)) % HashTableSize;
        if (HashTable[i].key == key) return BLANK;
        ++j;
    }
    return key;
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER T1 search_cuckoo(T1 key, size_t i, size_t j, cell<T1,T2>* HashTable1, cell<T1,T2>* HashTable2, size_t HashTableSize, hash_function function){
    size_t k = 1;

    if (HashTable1[i].key == key) return BLANK;
    if (HashTable2[j].key == key) return BLANK;
    
    while (k<HashTableSize){
        i = (i + k) % (2*HashTableSize);
        j = (j + k) % (2*HashTableSize);
        
        if (HashTable1[i].key == key) return BLANK;
        if (HashTable2[j].key == key) return BLANK;
        ++k;
    }
    return key;
};

/////////////////////////////////////////////////////////////////////////////////////////
//Löschung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
DEVICEQUALIFIER void delete_normal(T1 key, size_t i, cell<T1,T2>* HashTable){
    T1 prev = atomicCAS(&HashTable[i].key,key, BLANK);
    
    if (prev == BLANK){
    //if (HashTable[i].key == key){
        HashTable[i].key = BLANK;
        HashTable[i].value = BLANK;
    }
};

//Lineare Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void delete_linear(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j = 0;
    T1 prev;
    
    while(j<HashTableSize){
        i = (i+j)%HashTableSize;
        prev = atomicCAS(&HashTable[i].key,key, BLANK);

        if (prev == BLANK){
        //if (HashTable[i].key == key){
            HashTable[i].key = BLANK;
            HashTable[i].value = BLANK;
            break;
        }
        ++j;
    }
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void delete_quadratic(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize){
    size_t j = 0;
    T1 prev;
    
    while((j/2)<HashTableSize){
        i = ((size_t) ((int) i + getProbe2(j))) % HashTableSize;
        prev = atomicCAS(&HashTable[i].key,key, BLANK);

        if (prev == BLANK){
        //if (HashTable[i].key == key){
            HashTable[i].key = BLANK;
            HashTable[i].value = BLANK;
            break;
        }
        ++j;
    }
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void delete_double(T1 key, size_t i, cell<T1,T2>* HashTable, size_t HashTableSize, hash_function function){
    size_t j = 0;
    T1 prev;
        
    while((j/2)<HashTableSize){
        i = (i+getHashProbe<T1>(key,j,HashTableSize,function))%HashTableSize;
        prev = atomicCAS(&HashTable[i].key,key, BLANK); 

        if (prev == BLANK){
        //if (HashTable[i].key == key){{
            HashTable[i].key = BLANK;
            HashTable[i].value = BLANK;
            break;
        }
        ++j;
    }
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void delete_cuckoo(T1 key, size_t i, size_t j, cell<T1,T2>* HashTable1, cell<T1,T2>* HashTable2, size_t HashTableSize, hash_function function){
    size_t k = 1;
    T1 prev1, prev2;
    
    prev1 = atomicCAS(&HashTable1[i].key,key, BLANK); 

    if (prev1 == BLANK){
    //if (HashTable1[i].key == key){
        HashTable1[i].key = BLANK;
        HashTable1[i].value = BLANK;
        return;
    }

    prev2 = atomicCAS(&HashTable2[j].key,key, BLANK); 
    
    if (prev2 == BLANK){
    //if (HashTable2[j].key == key){
        HashTable2[j].key = BLANK;
        HashTable2[j].value = BLANK;
        return;
    }
    
    while (k<HashTableSize){
        i = (i + k) % (2*HashTableSize);
        j = (j + k) % (2*HashTableSize);
        
        prev1 = atomicCAS(&HashTable1[i].key,key, BLANK); 

        if (prev1 == BLANK){
        //if (HashTable1[i].key == key){
            HashTable1[i].key = BLANK;
            HashTable1[i].value = BLANK;
            break;
        }

        prev2 = atomicCAS(&HashTable2[j].key,key, BLANK); 

        if (prev2 == BLANK){
        //if (HashTable2[j].key == key){
            HashTable2[j].key = BLANK;
            HashTable2[j].value = BLANK;
            break;
        }
        ++k;
    }
};






















#endif