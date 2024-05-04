#ifndef HASH_METHODS_CUH
#define HASH_METHODS_CUH

#include <../include/base.h>
#include <../include/hash_function.cuh>

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von Zellen in einer oder bei ggf. Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void insert_normal(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j;
    T1 prev, key;
    T2 value;

    //2. Setze eine globale ID eines Threads
    i = blockIdx.x * blockDim.x + threadIdx.x;

    //3a. Lege einen Schlüssel und dessen Wert mit ihren Werten auf dem globalen Speicher fest 
    key = cells[i].key;
    value = cells[i].value;
    //3b. Warte, bis alle Werte von den Zellen auf die Variablen komplett übertragen werden
    __syncthreads();

    //4. Setze den Hashwert eines Schlüssels
    j = getHash<T2>(value,hashTableSize,function);

    //5. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
    prev = atomicCAS(&hashTable[j].key, BLANK, key);
    
    //6. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
    //   Belege bei einer freien Zelle die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und dessen Wert
    if (prev == BLANK || prev == key){
        hashTable[j].key = key;
        hashTable[j].value = value;
    }
};

//Lineares Sondieren
template <typename T1, typename T2>
__global__ void insert_linear(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T1 prev, key;
    T2 value;

    //2. Setze eine globale ID eines Threads
    i = blockIdx.x * blockDim.x + threadIdx.x;

    //3a. Lege einen Schlüssel und dessen Wert mit ihren Werten auf dem globalen Speicher fest    
    key = cells[i].key;
    value = cells[i].value;
    //3b. Warte, bis alle Werte von den Zellen auf die Variablen komplett übertragen werden
    __syncthreads();

    //4. Setze den Hashwert eines Schlüssels und den Anfangsindex einer Schleife
    j = getHash<T2>(value,hashTableSize,function);
    k = 0;

    //5. Führe einen Schleifendurchlauf aus, der die Größe einer Hashtabelle hat
    while(k < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch lineare Sondierungsfunktion
        j = (j + k) % hashTableSize;

        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        //    Belege bei einer freien Zelle die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und dessen Wert
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
};

//Quadratisches Sondieren
template <typename T1, typename T2>
__global__ void insert_quadratic(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T1 prev, key;
    T2 value;

    //2. Setze eine globale ID eines Threads
    i = blockIdx.x * blockDim.x + threadIdx.x;

    //3a. Lege einen Schlüssel und dessen Wert mit ihren Werten auf dem globalen Speicher fest
    key = cells[i].key;
    value = cells[i].value;
    //3b. Warte, bis alle Werte von den Zellen auf die Variablen komplett übertragen werden
    __syncthreads();

    //4. Setze den Hashwert eines Schlüssels und den Anfangsindex einer Schleife
    j = getHash<T2>(value,hashTableSize,function);
    k = 0;
    
    //5. Führe einen Schleifendurchlauf aus, die die doppelte Größe einer Hashtabelle hat
    while((k/2) < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch quadratische Sondierungsfunktion
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;

        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle
        prev = atomicCAS(&hashTable[j].key, BLANK, key);

        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        //   Belege bei einer freien Zelle die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und dessen Wert
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void insert_double(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    //1. Deklariere die Variablen
    size_t i, j, k;
    T1 prev, key;
    T2 value;

    //2. Setze eine globale ID eines Threads
    i = blockIdx.x * blockDim.x + threadIdx.x;

    //3a. Lege einen Schlüssel und dessen Wert mit ihren Werten auf dem globalen Speicher fest
    key = cells[i].key;
    value = cells[i].value;
    //3b. Warte, bis alle Werte von den Zellen auf die Variablen komplett übertragen werden
    __syncthreads();

    //4. Setze den Hashwert eines Schlüssels und den Anfangsindex einer Schleife
    j = getHash<T2>(value,hashTableSize,function1);
    k = 0; 

    //5. Führe einen Schleifendurchlauf aus, die die Größe einer Hashtabelle hat
    while(k < hashTableSize){
        //5a. Berechne den neuen Hashwert eines Schlüssels durch eine neue Hashfunktion
        j = (j + getHashProbe<T2>(value,k,hashTableSize,function2)) % hashTableSize;
        //5b. Vertausche einen Schlüssel mit dem anderen in einer Hashtabelle 
        prev = atomicCAS(&hashTable[j].key, BLANK, key);
        
        //5c. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        //   Belege bei einer freien Zelle die Zelle in der Hashtabelle mit neuen Werten vom Schlüssel und dessen Wert
        if (prev == BLANK || prev == key){
            hashTable[j].key = key;
            hashTable[j].value = value;
            break;
        }
        //5d. Erhöhe den Hashwert eines Schlüssels
        ++k;
    }
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void insert_cuckoo(cell<T1,T2> * cells, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    //1. Deklariere die Variablen
    size_t i, j, k, m, max_hash_table_size;
    T1 prev1, prev2, key;
    T2 value;
    
    //2. Setze eine globale ID eines Threads
    i = blockIdx.x * blockDim.x + threadIdx.x;

    //3a. Lege einen Schlüssel und dessen Wert mit ihren Werten auf dem globalen Speicher fest
    key = cells[i].key;
    value = cells[i].value;
    //3b. Warte, bis alle Werte von den Zellen auf die Variablen komplett übertragen werden
    __syncthreads();

    //4. Setze die Hashwerte eines Schlüssels, den Anfangsindex einer Schleife und die maximale Anzahl an Schleifen
    j = getHash<T2>(value,hashTableSize,function1);
    k = getHash<T2>(value,hashTableSize,function2);
    m = 0; 
    max_hash_table_size = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*hashTableSize);
    
    //5. Überprüfe, ob der Schlüssel in einer Zelle der ersten oder zweiten Hashtabelle vorhanden ist
    //   Falls der Schlüssel vorhanden ist, beende Cuckoo-Hashverfahren,
    //   sonst, führe Cuckoo-Hashverfahren aus
    if (key == hashTable1[j].key || key == hashTable2[k].key) return;

    //6. Führe einen Schleifendurchlauf aus, die die maximale festgelegte Anzahl an Schleifen hat
    while (m < max_hash_table_size){
        //6a. Berechne die Hashwerte eines Schlüssels durch zwei lineare Sondierungsfunktionen
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
        //6b. Vertausche einen Schlüssel mit dem anderen in der ersten Hashtabelle
        swapCells<T1,T2>(key,value,j,hashTable1);
        prev1 = atomicCAS(&hashTable1[j].key, key, BLANK);
        
        //6c. Überprüfe, ob der Schlüssel gegen einen anderen in der ersten Hashtabelle ausgetauscht wird
        //    Falls ja, verlasse die Schleife
        //    Sonst, setze fort.
        if (prev1 == BLANK) break;
        
        //6d. Vertausche einen Schlüssel mit dem anderen in der zweiten Hashtabelle
        swapCells<T1,T2>(key,value,k,hashTable2);
        prev2 = atomicCAS(&hashTable2[k].key, key, BLANK);

        //6e. Überprüfe, ob der Schlüssel gegen einen anderen in der zweiten Hashtabelle ausgetauscht wird
        //    Falls ja, verlasse die Schleife
        //    Sonst, setze fort.
        if (prev2 == BLANK) break;
    
        //6f. Erhöhe den Hashwert eines Schlüssels
        ++m;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Suche nach einer Liste von Schlüsseln in einer oder ggf. bei Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void search_normal(cell<T1,T2> * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedCells[h].key = keyList[i].key;
    sharedCells[h].value = keyList[i].value;

    __syncthreads();

    j = getHash<T2>(sharedCells[h].value,hashTableSize,function);
    
    if (hashTable[j].key == sharedCells[h].key){
        keyListResult[i] = BLANK;
    }else{
        keyListResult[i] = sharedCells[h].key;
    }
};

//Lineares Sondieren
template <typename T1, typename T2>
__global__ void search_linear(cell<T1,T2> * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedCells[h].key = keyList[i].key;
    sharedCells[h].value = keyList[i].value;

    __syncthreads();
    
    j = getHash<T2>(sharedCells[h].value,hashTableSize,function);
    k = 0;

    keyListResult[i] = sharedCells[h].key;

    while(k < hashTableSize){
        j = (j + k) % hashTableSize;
        if (hashTable[j].key == sharedCells[h].key){
            keyListResult[i] = BLANK;
            break;
        } 
        ++k;
    }
};

//Quadratisches Sondieren
template <typename T1, typename T2>
__global__ void search_quadratic(cell<T1,T2> * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedCells[h].key = keyList[i].key;
    sharedCells[h].value = keyList[i].value;

    __syncthreads();

    j = getHash<T2>(sharedCells[h].value,hashTableSize,function);
    k = 0;

    keyListResult[i] = sharedCells[h].key;

    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        if (hashTable[j].key == sharedCells[h].key){
            keyListResult[i] = BLANK;
            break;
        }
        ++k;
    }
};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void search_double(cell<T1,T2> * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = keyList[i].key;
    sharedCells[h].value = keyList[i].value;

    __syncthreads();

    j = getHash<T2>(sharedCells[h].value,hashTableSize,function1);
    k = 0;

    keyListResult[i] = sharedCells[h].key;
    
    while(k < hashTableSize){
        j = (j + getHashProbe<T2>(sharedCells[h].value,k,hashTableSize,function2)) % hashTableSize;
        if (hashTable[j].key == sharedCells[h].key){
            keyListResult[i] = BLANK;
            break;
        } 
        ++k;
    }
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void search_cuckoo(cell<T1,T2>* keyList, T1 * keyListResult, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k, m;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = keyList[i].key;
    sharedCells[h].value = keyList[i].value;

    __syncthreads();

    j = getHash<T2>(sharedCells[h].value,hashTableSize,function1);
    k = getHash<T2>(sharedCells[h].value,hashTableSize,function2);
    m = 1;

    keyListResult[i] = sharedCells[h].key;
    
    if (hashTable1[j].key == sharedCells[h].key){
        keyListResult[i] = BLANK;
        return;
    }
    if (hashTable2[k].key == sharedCells[h].key){
        keyListResult[i] = BLANK;
        return;
    }
    
    while (m < hashTableSize){
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
        if (hashTable1[j].key == sharedCells[h].key){
            keyListResult[i] = BLANK;
            return;
        } 

        if (hashTable2[k].key == sharedCells[h].key){
            keyListResult[i] = BLANK;
            return;
        }
        ++m;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Löschung von Zellen in einer oder ggf. bei Cuckoo-Hashverfahren zwei Hashtabelle(n)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Keine Kollisionsauflösung
template <typename T1, typename T2>
__global__ void delete_normal(cell<T1,T2> * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j;
    T1 key, prev;
    T2 value;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = BLANK;
    sharedCells[h].value = BLANK;

    __syncthreads();

    key = keyList[i].value;
    value = keyList[i].key;

    j = getHash<T2>(value,hashTableSize,function);
    
    prev = atomicCAS(&hashTable[j].key, key, sharedCells[h].key);
    
    if (prev == sharedCells[h].key){
        hashTable[j].key = sharedCells[h].key;
        hashTable[j].value = sharedCells[h].value;
    }
};

//Lineares Sondieren
template <typename T1, typename T2>
__global__ void delete_linear(cell<T1,T2> * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    T1 key, prev;
    T2 value;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = BLANK;
    sharedCells[h].value = BLANK;

    __syncthreads();

    key = keyList[i].key;
    value = keyList[i].value;

    j = getHash<T2>(value,hashTableSize,function);
    k = 0;

    while(k < hashTableSize){
        j = (j+k) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, key, sharedCells[h].key);

        if (prev == sharedCells[h].key){
            hashTable[j].key = sharedCells[h].key;
            hashTable[j].value = sharedCells[h].value;
            break;
        }
        ++k;
    }
};

//Quadratisches Sondieren
template <typename T1, typename T2>
__global__ void delete_quadratic(cell<T1,T2> * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    T1 key, prev;
    T2 value;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = BLANK;
    sharedCells[h].value = BLANK;

    __syncthreads();

    key = keyList[i].key;
    value = keyList[i].value;
    
    j = getHash<T2>(value,hashTableSize,function);
    k = 0;

    while((k/2) < hashTableSize){
        j = ((size_t) ((int) j + getProbe2(k))) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, key, sharedCells[h].key);

        if (prev == sharedCells[h].key){
            hashTable[j].key = sharedCells[h].key;
            hashTable[j].value = sharedCells[h].value;
            break;
        }
        ++k;
    }
};

//Doppelte Hashverfahren
template <typename T1, typename T2>
__global__ void delete_double(cell<T1,T2> * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k;
    T1 key, prev;
    T2 value;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = BLANK;
    sharedCells[h].value = BLANK;

    __syncthreads();

    key = keyList[i].key;
    value = keyList[i].value;
    
    j = getHash<T2>(value,hashTableSize,function1);
    k = 0;
    
    while((k/2) < hashTableSize){
        j = (j+getHashProbe<T2>(value,k,hashTableSize,function2)) % hashTableSize;
        prev = atomicCAS(&hashTable[j].key, key, sharedCells[h].key); 

        if (prev == sharedCells[h].key){
            hashTable[j].key = sharedCells[h].key;
            hashTable[j].value =  sharedCells[h].value;
            break;
        }
        ++k;
    }
};

//Cuckoo-Hashverfahren
template <typename T1, typename T2>
__global__ void delete_cuckoo(cell<T1,T2> * keyList, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    extern __shared__ cell<T1,T2> sharedCells[];
    size_t h, i, j, k, m;
    T1 key, prev1, prev2;
    T2 value;
    
    h = threadIdx.x;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCells[h].key = BLANK;
    sharedCells[h].value = BLANK;

    __syncthreads();

    key = keyList[i].key;
    value = keyList[i].value;

    j = getHash<T2>(value,hashTableSize,function1);
    k = getHash<T2>(value,hashTableSize,function2);
    m = 1;
    
    prev1 = atomicCAS(&hashTable1[j].key, key, sharedCells[h].key); 

    if (prev1 == sharedCells[h].key){
        hashTable1[j].key = sharedCells[h].key;
        hashTable1[j].value = sharedCells[h].value;
        return;
    }

    prev2 = atomicCAS(&hashTable2[k].key, key, sharedCells[h].key); 
    
    if (prev2 == sharedCells[h].key){
        hashTable2[k].key = sharedCells[h].key;
        hashTable2[k].value = sharedCells[h].value;
        return;
    }
    
    while (m < hashTableSize){
        j = (j + m) % hashTableSize;
        k = (k + m) % hashTableSize;
        
        prev1 = atomicCAS(&hashTable1[j].key, key, sharedCells[h].key); 

        if (prev1 == sharedCells[h].key){
            hashTable1[j].key = sharedCells[h].key;
            hashTable1[j].value = sharedCells[h].value;
            break;
        }

        prev2 = atomicCAS(&hashTable2[k].key, key, sharedCells[h].key); 

        if (prev2 == sharedCells[h].key){
            hashTable2[k].key = sharedCells[h].key;
            hashTable2[k].value = sharedCells[h].value;
            break;
        }
        ++m;
    }
};

#endif