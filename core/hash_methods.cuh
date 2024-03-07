#ifndef HASH_METHODS_CUH
#define HASH_METHODS_CUH

//#include <stdio.h>
#include <stdint.h>

#include <../include/base.h>
#include <../include/hash_function.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void insert_normal(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key, prev;
    T2 value; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    __syncthreads();
    
    prev = atomicCAS(&hashTable[j].key, BLANK, key);
    
    if (prev == BLANK || prev == key){
        hashTable[j].key = key;
        hashTable[j].value = value;
        __syncthreads();
        return;
    }

    __syncthreads();
};

//Lineare Hashverfahren
template <typename T1, typename T2>
__global__ void insert_linear(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    T2 value; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    k = 0;

    __syncthreads();

    while(k<hashTableSize){
        j = (j + k) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        ++k;
    }

    __syncthreads();
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
__global__ void insert_quadratic(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    T2 value; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    k = 0;

    __syncthreads();
    
    while(k<hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        ++k;
    }
    
    __syncthreads();
};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void insert_double(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function1);

    k = 0; 

    __syncthreads();

    while(k<hashTableSize){
        j = (j + getHashProbe(key,k,hashTableSize,function2)) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void insert_cuckoo(cell<T1,T2> * cells, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m, max_hash_table_size;
    T1 key, prev1, prev2;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);

    m = 1; 
    max_hash_table_size = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*hashTableSize);

    __syncthreads();

    prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);
    
    if (prev1 == BLANK || prev1 == key){
        hashTable1[j].key = key;
        hashTable1[j].value = value;
        return;
    }

    prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

    if (prev2 == BLANK || prev2 == key){
        hashTable2[k].key = key;
        hashTable2[k].value = value;
        return;
    }
    
    while (m<max_hash_table_size){
        j = (j + m) % (2*hashTableSize);
        k = (k + m) % (2*hashTableSize);
        
        //Vertausche zwei Schlüssel innerhalb der 1. Hashtabelle
        swapCells<T1,T2>(key,value,j,hashTable1);

        prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);
        
        if (prev1 == BLANK || prev1 == key){
            hashTable1[j].key = key;
            hashTable1[j].value = value;
            break;
        }

        //Vertausche zwei Schlüssel innerhalb der 2. Hashtabelle
        swapCells<T1,T2>(key,value,k,hashTable2);

        prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

        if (prev2 == BLANK || prev2 == key){
            hashTable2[k].key = key;
            hashTable2[k].value = value;
            break;
        }

        ++m;
    }

    __syncthreads();

};

/////////////////////////////////////////////////////////////////////////////////////////
//Suchen nach einer Liste von Schlüsseln in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void search_normal(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);
    
    __syncthreads();
    
    if (hashTable[j].key == key){
        keyListResult[i] = BLANK;
    }else{
        keyListResult[i] = key;
    }

    __syncthreads();

};

//Lineare Hashverfahren
template <typename T1, typename T2>
__global__ void search_linear(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    
    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);
    k = 0;

    __syncthreads();

    keyListResult[i] = key;

    while(k < hashTableSize){
        j = (j + k) % hashTableSize;
        if (hashTable[j].key == key){
            keyListResult[i] = BLANK;
            break;
        } 
        ++k;
    }

    __syncthreads();
};

//Quadratische Hashverfahren
template <typename T1, typename T2>
__global__ void search_quadratic(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    
    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);
    k = 0;

    __syncthreads();

    keyListResult[i] = key;

    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        if (hashTable[j].key == key){
            keyListResult[i] = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void search_double(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);
    k = 0;

    __syncthreads();

    keyListResult[i] = key;
    
    while(k < hashTableSize){
        j = (j + getHashProbe(key,k,hashTableSize,function2)) % hashTableSize;
        if (hashTable[j].key == key){
            keyListResult[i] = BLANK;
            break;
        } 
        ++k;
    }

    __syncthreads();
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void search_cuckoo(T1* keyList, T1 * keyListResult, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);
    m = 1;

    __syncthreads();

    keyListResult[i] = key;
    
    if (hashTable1[j].key == key){
        keyListResult[i] = BLANK;
        __syncthreads();
        return;
    }
    if (hashTable2[k].key == key){
        keyListResult[i] = BLANK;
        __syncthreads();
        return;
    }
    
    while (m < hashTableSize){
        j = (j + m) % (2*hashTableSize);
        k = (k + m) % (2*hashTableSize);
        
        if (hashTable1[j].key == key){
            keyListResult[i] = BLANK;
            __syncthreads();
            return;
        } 

        if (hashTable2[k].key == key){
            keyListResult[i] = BLANK;
            __syncthreads();
            return;
        }
        ++m;
    }

    __syncthreads();
};

/////////////////////////////////////////////////////////////////////////////////////////
//Löschung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void delete_normal(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);

    __syncthreads();
    
    prev = atomicCAS(&hashTable[j].key,key, BLANK);
    
    if (prev == BLANK){
        hashTable[j].key = BLANK;
        hashTable[j].value = BLANK;
    }

    __syncthreads();

};

//Lineare Hashverfahren
template <typename T1, typename T2>
__global__ void delete_linear(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);
    k = 0;

    __syncthreads();

    while(k < hashTableSize){
        j = (j+k) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key,key, BLANK);

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].value = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Quadratische Hashverfahren
template <typename T1, typename T2>
__global__ void delete_quadratic(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);
    k = 0;

    __syncthreads();
    
    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key,key, BLANK);

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].value = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void delete_double(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);
    k = 0;

    __syncthreads();
    
    while((k/2) < hashTableSize){
        j = (j+getHashProbe<T1>(key,k,hashTableSize,function2)) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key,key, BLANK); 

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].value = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void delete_cuckoo(T1 * keyList, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m;
    T1 key, prev1, prev2;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);
    m = 1;

    __syncthreads();
    
    prev1 = atomicCAS(&hashTable1[j].key,key, BLANK); 

    if (prev1 == BLANK){
        hashTable1[j].key = BLANK;
        hashTable1[j].value = BLANK;
        __syncthreads();
        return;
    }

    prev2 = atomicCAS(&hashTable2[k].key,key, BLANK); 
    
    if (prev2 == BLANK){
        hashTable2[k].key = BLANK;
        hashTable2[k].value = BLANK;
        __syncthreads();
        return;
    }
    
    while (m < hashTableSize){
        j = (j + m) % (2*hashTableSize);
        k = (k + m) % (2*hashTableSize);
        
        prev1 = atomicCAS(&hashTable1[j].key,key, BLANK); 

        if (prev1 == BLANK){
            hashTable1[j].key = BLANK;
            hashTable1[j].value = BLANK;
            break;
        }

        prev2 = atomicCAS(&hashTable2[k].key,key, BLANK); 

        if (prev2 == BLANK){
            hashTable2[k].key = BLANK;
            hashTable2[k].value = BLANK;
            break;
        }
        ++m;
    }

    __syncthreads();
};

#endif