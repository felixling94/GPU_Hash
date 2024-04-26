#ifndef HASH_METHODS_CUH
#define HASH_METHODS_CUH

#include <../include/base.h>
#include <../include/hash_function.cuh>

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von Zellen in einer oder bei ggf. Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T>
__global__ void insert_normal(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j;
    T key, prev, key_length; 

    //2. Setze die globale ID eines Threads, und einen Schlüssel und seinen Längen
    i = blockIdx.x * blockDim.x + threadIdx.x;
    key = cells[i].key;
    key_length = cells[i].key_length;

    //3. Setze den Hashwert eines Schlüssels
    j = getHash<T>(key_length,hashTableSize,function);

    //4. Synchronisiere alle Threads
    __syncthreads();
    
    //5. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
    prev = atomicCAS(&hashTable[j].key, BLANK, key);
    
    //6a. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
    if (prev == BLANK || prev == key){
        //6a1. Belege die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
        hashTable[j].key = key;
        hashTable[j].key_length = key_length;
        //6a2. Synchronisiere alle Threads
        __syncthreads();
        return;
    }
    //6b. Synchronisiere alle Threads
    __syncthreads();
};

//Lineares Sondieren
template <typename T>
__global__ void insert_linear(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T key, prev, key_length; 

    //2. Setze die globale ID eines Threads, und einen Schlüssel und seinen Längen
    i = blockIdx.x * blockDim.x + threadIdx.x;
    key = cells[i].key;
    key_length = cells[i].key_length;
    
    //3. Setze den Hashwert eines Schlüssels und den Index einer Schleife
    j = getHash<T>(key_length,hashTableSize,function);
    k = 0;

    //4. Synchronisiere alle Threads
    __syncthreads();

    //5. Führe einen Schleifendurchlauf durch die Größe einer Hashtabelle aus
    while(k < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch lineare Sondierungsfunktion
        j = (j + k) % hashTableSize;
        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        if (prev == BLANK || prev == key){
            //5c1. Belege die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
    //6. Synchronisiere alle Threads
    __syncthreads();
};

//Quadratisches Sondieren
template <typename T>
__global__ void insert_quadratic(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T key, prev, key_length; 

    //2. Setze die globale ID eines Threads, und einen Schlüssel und seinen Längen
    i = blockIdx.x * blockDim.x + threadIdx.x;
    key = cells[i].key;
    key_length = cells[i].key_length;

    //3. Setze den Hashwert eines Schlüssels und den Index einer Schleife
    j = getHash<T>(key_length,hashTableSize,function);
    k = 0;

    //4. Synchronisiere alle Threads
    __syncthreads();
    
    //5. Führe einen Schleifendurchlauf durch die doppelte Größe einer Hashtabelle aus
    while((k/2) < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch quadratische Sondierungsfunktion
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
        prev = atomicCAS(&hashTable[j].key, BLANK, key);

        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        if (prev == BLANK || prev == key){
            //5c1. Belege die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
    //6. Synchronisiere alle Threads
    __syncthreads();
};

//Doppelte Hashverfahren
template <typename T>
__global__ void insert_double(cell<T> * cells, cell<T> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T key, prev, key_length;
    
    //2. Setze die globale ID eines Threads, und einen Schlüssel und seinen Längen
    i = blockIdx.x * blockDim.x + threadIdx.x;
    key = cells[i].key;
    key_length = cells[i].key_length;

    //3. Setze den Hashwert eines Schlüssels und den Index einer Schleife
    j = getHash<T>(key_length,hashTableSize,function1);
    k = 0; 

    //4. Synchronisiere alle Threads
    __syncthreads();

    //5. Führe einen Schleifendurchlauf durch die Größe einer Hashtabelle aus
    while(k < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch eine neue Hashfunktion
        j = (j + getHashProbe<T>(key_length,k,hashTableSize,function2)) % hashTableSize;
        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle 
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        if (prev == BLANK || prev == key){
            //5c1. Belege die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
            hashTable[j].key = key;
            hashTable[j].key_length = key_length;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
    //6. Synchronisiere alle Threads
    __syncthreads();
};

//Cuckoo-Hashverfahren
template <typename T>
__global__ void insert_cuckoo(cell<T> * cells, cell<T> * hashTable1, cell<T> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    //1. Deklariere die Variablen
    size_t i, j, k, m, max_hash_table_size;
    T key, prev1, prev2, key_length;
    
    //2. Setze die globale ID eines Threads, und einen Schlüssel und seinen Längen
    i = blockIdx.x * blockDim.x + threadIdx.x;
    key = cells[i].key;
    key_length = cells[i].key_length;

    //3. Setze die Hashwerte eines Schlüssels, den Index einer Schleife und die maximale Anzahl an Schleifen
    j = getHash<T>(key_length,hashTableSize,function1);
    k = getHash<T>(key_length,hashTableSize,function2);
    m = 1; 
    max_hash_table_size = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*hashTableSize);

    //4. Synchronisiere alle Threads
    __syncthreads();

    //5a1. Vertausche einen Schlüssel mit dem anderen in der ersten Hashtabelle
    prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);

    //5a2. Überprüfe, ob die Zelle in der ersten Hashtabelle belegt ist
    if (prev1 == BLANK || prev1 == key){
        //5a2a. Belege die Zelle in der ersten Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
        hashTable1[j].key = key;
        hashTable1[j].key_length = key_length;
        //5a2b. Synchronisiere alle Threads
        __syncthreads();
        return;
    }

    //5b1. Vertausche einen Schlüssel mit dem anderen in der zweiten Hashtabelle
    prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

    //5b2. Überprüfe, ob die Zelle in der zweiten Hashtabelle belegt ist
    if (prev2 == BLANK || prev2 == key){
        //5b2a. Belege die Zelle in der zweiten Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
        hashTable2[k].key = key;
        hashTable2[k].key_length = key_length;
        //5b2b. Synchronisiere alle Threads
        __syncthreads();
        return;
    }
    
    //6. Führe einen Schleifendurchlauf durch die maximale festgelegte Anzahl an Schleifen aus
    while (m < max_hash_table_size){
        //6a. Berechne die Hashwerte eines Schlüssels durch zwei lineare Sondierungsfunktionen
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
        //6b. Vertausche einen Schlüssel mit dem anderen in der ersten Hashtabelle
        swapCells<T>(key,key_length,j,hashTable1);
        prev1 = atomicCAS(&hashTable1[j].key, BLANK, key);
        
        //6c. Überprüfe, ob die Zelle in der ersten Hashtabelle belegt ist
        if (prev1 == BLANK || prev1 == key){
            //6c1. Belege die Zelle in der ersten Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
            hashTable1[j].key = key;
            hashTable1[j].key_length = key_length;
            break;
        }

        //6d. Vertausche einen Schlüssel mit dem anderen in der zweiten Hashtabelle
        swapCells<T>(key,key_length,k,hashTable2);
        prev2 = atomicCAS(&hashTable2[k].key, BLANK, key);

        //6e. Überprüfe, ob die Zelle in der zweiten Hashtabelle belegt ist
        if (prev2 == BLANK || prev2 == key){
            //6e1. Belege die Zelle in der zweiten Hashtabelle mit neuen Werten vom Schlüssel und deren Wert
            hashTable2[k].key = key;
            hashTable2[k].key_length = key_length;
            break;
        }
        //6f. Erhöhe den Hashwert eines Schlüssels
        ++m;
    }
    //7. Synchronisiere alle Threads
    __syncthreads();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Suche nach einer Liste von Schlüsseln in einer oder ggf. bei Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
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

//Lineares Sondieren
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

//Quadratisches Sondieren
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
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Löschung von Zellen in einer oder ggf. bei Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
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

//Lineares Sondieren
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

//Quadratisches Sondieren
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
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
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