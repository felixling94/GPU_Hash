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
template <typename T>
__global__ void insert_normal(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T key, prev, key_length; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    key_length = cells[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);

    __syncthreads();
    
    prev = atomicCAS(&hashTable[j].key, BLANK, key);
    
    if (prev == BLANK || prev == key){
        hashTable[j].key = key;
        hashTable[j].key_length = key_length;
        __syncthreads();
        return;
    }

    __syncthreads();
};

//Lineare Hashverfahren
template <typename T>
__global__ void insert_linear(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, prev, key_length; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    key_length = cells[i].key_length;
    
    j = getHash<T>(key_length,hashTableSize,function);

    k = 0;

    __syncthreads();

    while(k < hashTableSize){
        j = (j + k) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        ++k;
    }

    __syncthreads();
};

//Quadratische Hashverfahren
template <typename T>
__global__ void insert_quadratic(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, prev, key_length; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    key_length = cells[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);

    k = 0;

    __syncthreads();
    
    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        ++k;
    }
    
    __syncthreads();
};

//Doppelte Hashverfahren
template <typename T>
__global__ void insert_double(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, prev, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    key_length = cells[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);

    k = 0; 

    __syncthreads();

    while(k < hashTableSize){
        j = (j + getHashProbe<T>(key_length,k,hashTableSize,function2)) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Cuckoo-Hashverfahren
template <typename T>
__global__ void insert_cuckoo(cell<T> * cells, cell<T> * hashTable1, cell<T> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m, max_hash_table_size;
    T key, prev1, prev2, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    key_length = cells[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);
    k = getHash<T>(key_length,hashTableSize,function2);

    m = 1; 
    max_hash_table_size = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*hashTableSize);

    __syncthreads();

    prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);
    
    if (prev1 == BLANK || prev1 == key){
        hashTable1[j].key = key;
        hashTable1[j].key_length = key_length;
        return;
    }

    prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

    if (prev2 == BLANK || prev2 == key){
        hashTable2[k].key = key;
        hashTable2[k].key_length = key_length;
        return;
    }
    
    while (m < max_hash_table_size){
        j = (j + m) % (2*hashTableSize);
        k = (k + m) % (2*hashTableSize);
        
        //Vertausche zwei Schlüssel innerhalb der 1. Hashtabelle
        swapCells<T>(key,key_length,j,hashTable1);

        prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);
        
        if (prev1 == BLANK || prev1 == key){
            hashTable1[j].key = key;
            hashTable1[j].key_length = key_length;
            break;
        }

        //Vertausche zwei Schlüssel innerhalb der 2. Hashtabelle
        swapCells<T>(key,key_length,k,hashTable2);

        prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

        if (prev2 == BLANK || prev2 == key){
            hashTable2[k].key = key;
            hashTable2[k].key_length = key_length;
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
template <typename T>
__global__ void search_normal(cell<T> * keyList, T * keyListResult, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T key, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);
    
    __syncthreads();
    
    if (hashTable[j].key == key){
        keyListResult[i] = BLANK;
    }else{
        keyListResult[i] = key;
    }

    __syncthreads();

};

//Lineare Hashverfahren
template <typename T>
__global__ void search_linear(cell<T> * keyList, T * keyListResult, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    
    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);
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
template <typename T>
__global__ void search_quadratic(cell<T> * keyList, T * keyListResult, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    
    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);
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
template <typename T>
__global__ void search_double(cell<T> * keyList, T * keyListResult, cell<T> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);
    k = 0;

    __syncthreads();

    keyListResult[i] = key;
    
    while(k < hashTableSize){
        j = (j + getHashProbe(key_length,k,hashTableSize,function2)) % hashTableSize;
        if (hashTable[j].key == key){
            keyListResult[i] = BLANK;
            break;
        } 
        ++k;
    }

    __syncthreads();
};

//Cuckoo-Hashverfahren
template <typename T>
__global__ void search_cuckoo(cell<T>* keyList, T * keyListResult, cell<T> * hashTable1, cell<T> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m;
    T key, key_length;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);
    k = getHash<T>(key_length,hashTableSize,function2);
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
template <typename T>
__global__ void delete_normal(cell<T> * keyList, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T key, key_length, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);

    __syncthreads();
    
    prev = atomicCAS(&hashTable[j].key,key, BLANK);
    
    if (prev == BLANK){
        hashTable[j].key = BLANK;
        hashTable[j].key_length = BLANK;
    }

    __syncthreads();

};

//Lineare Hashverfahren
template <typename T>
__global__ void delete_linear(cell<T> * keyList, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);
    k = 0;

    __syncthreads();

    while(k < hashTableSize){
        j = (j+k) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key,key, BLANK);

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].key_length = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Quadratische Hashverfahren
template <typename T>
__global__ void delete_quadratic(cell<T> * keyList, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function);
    k = 0;

    __syncthreads();
    
    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key,key, BLANK);

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].key_length = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();

};

//Doppelte Hashverfahren
template <typename T>
__global__ void delete_double(cell<T> * keyList, cell<T> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T key, key_length, prev;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);
    k = 0;

    __syncthreads();
    
    while((k/2) < hashTableSize){
        j = (j+getHashProbe<T>(key_length,k,hashTableSize,function2)) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, key, BLANK); 

        if (prev == BLANK){
            hashTable[j].key = BLANK;
            hashTable[j].key_length = BLANK;
            break;
        }
        ++k;
    }

    __syncthreads();
};

//Cuckoo-Hashverfahren
template <typename T>
__global__ void delete_cuckoo(cell<T> * keyList, cell<T> * hashTable1, cell<T> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k, m;
    T key, key_length, prev1, prev2;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i].key;
    key_length = keyList[i].key_length;

    j = getHash<T>(key_length,hashTableSize,function1);
    k = getHash<T>(key_length,hashTableSize,function2);
    m = 1;

    __syncthreads();
    
    prev1 = atomicCAS(&hashTable1[j].key,key, BLANK); 

    if (prev1 == BLANK){
        hashTable1[j].key = BLANK;
        hashTable1[j].key_length = BLANK;
        __syncthreads();
        return;
    }

    prev2 = atomicCAS(&hashTable2[k].key,key, BLANK); 
    
    if (prev2 == BLANK){
        hashTable2[k].key = BLANK;
        hashTable2[k].key_length = BLANK;
        __syncthreads();
        return;
    }
    
    while (m < hashTableSize){
        j = (j + m) % (2*hashTableSize);
        k = (k + m) % (2*hashTableSize);
        
        prev1 = atomicCAS(&hashTable1[j].key,key, BLANK); 

        if (prev1 == BLANK){
            hashTable1[j].key = BLANK;
            hashTable1[j].key_length = BLANK;
            break;
        }

        prev2 = atomicCAS(&hashTable2[k].key,key, BLANK); 

        if (prev2 == BLANK){
            hashTable2[k].key = BLANK;
            hashTable2[k].key_length = BLANK;
            break;
        }
        ++m;
    }

    __syncthreads();
};

#endif