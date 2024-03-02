#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../include/declaration.cuh>
#include <../include/hash_function.cuh>
#include <../core/hash_method.cuh>
#include <../tools/timer.cuh>
#include <../tools/benchmark.h>

/////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
GLOBALQUALIFIER void insert_normal_kernel(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    T2 value; 

    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    insert_normal(key, value, j, hashTable);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert_linear_kernel(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    insert_linear<T1,T2>(key, value, j, hashTable, hashTableSize);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert_quadratic_kernel(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function);

    insert_quadratic<T1,T2>(key, value, j, hashTable, hashTableSize);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert_double_kernel(cell<T1,T2> * cells, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function1);

    insert_double<T1,T2>(key, value, j, hashTable, hashTableSize,function2);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert_cuckoo_kernel(cell<T1,T2> * cells, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    T2 value;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = cells[i].key;
    value = cells[i].value;
    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);
    
    insert_cuckoo<T1,T2>(key, value, j, k, hashTable1, hashTable2, hashTableSize,function2);
};

/////////////////////////////////////////////////////////////////////////////////////////
//Suchen nach einer Liste von Schlüsseln in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
GLOBALQUALIFIER void search_normal_kernel(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];

    __syncthreads();

    j = getHash<T1>(key,hashTableSize,function);

    keyListResult[i] = search_normal(key, j, hashTable);

    __syncthreads();
};

template <typename T1, typename T2>
GLOBALQUALIFIER void search_linear_kernel(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];

    __syncthreads();

    j = getHash<T1>(key,hashTableSize,function);

    keyListResult[i] = search_linear(key, j, hashTable,hashTableSize);

    __syncthreads();
};

template <typename T1, typename T2>
GLOBALQUALIFIER void search_quadratic_kernel(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];

    __syncthreads();

    j = getHash<T1>(key,hashTableSize,function);

    keyListResult[i] = search_quadratic(key, j, hashTable,hashTableSize);

    __syncthreads();
};

template <typename T1, typename T2>
GLOBALQUALIFIER void search_double_kernel(T1 * keyList, T1 * keyListResult, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];

    __syncthreads();

    j = getHash<T1>(key,hashTableSize,function1);

    keyListResult[i] = search_double<T1,T2>(key, j, hashTable, hashTableSize,function2);

    __syncthreads();
};

template <typename T1, typename T2>
GLOBALQUALIFIER void search_cuckoo_kernel(T1* keyList, T1 * keyListResult, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];

    __syncthreads();

    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);
    
    keyListResult[i] = search_cuckoo<T1,T2>(key, j, k, hashTable1, hashTable2, hashTableSize,function2);

    __syncthreads();
};

/////////////////////////////////////////////////////////////////////////////////////////
//Löschung von Zellen in einer Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
GLOBALQUALIFIER void delete_normal_kernel(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);

    delete_normal(key, j, hashTable);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void delete_linear_kernel(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);

    delete_linear<T1,T2>(key, j, hashTable, hashTableSize);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void delete_quadratic_kernel(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function);

    delete_quadratic<T1,T2>(key, j, hashTable, hashTableSize);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void delete_double_kernel(T1 * keyList, cell<T1,T2> * hashTable, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);

    delete_double<T1,T2>(key, j, hashTable, hashTableSize,function2);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void delete_cuckoo_kernel(T1 * keyList, cell<T1,T2> * hashTable1, cell<T1,T2> * hashTable2, size_t hashTableSize, hash_function function1, hash_function function2){
    int i_inBlock, blockID, i;
    size_t j, k;
    T1 key;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    key = keyList[i];
    j = getHash<T1>(key,hashTableSize,function1);
    k = getHash<T1>(key,hashTableSize,function2);
    
    delete_cuckoo<T1,T2>(key, j, k, hashTable1, hashTable2, hashTableSize,function2);
};

template <typename T1, typename T2>
Hash_Table<T1,T2>::Hash_Table():type_hash(no_probe),function1(modulo),function2(modulo),table_size(2){
    hash_table1 = new cell<T1,T2>[2];
};

template <typename T1, typename T2>
Hash_Table<T1,T2>::Hash_Table(hash_type HashType, hash_function function_1, hash_function function_2, size_t TableSize):
type_hash(HashType),function1(function_1),function2(function_2),table_size(TableSize){
    hash_table1 = new cell<T1,T2>[TableSize];
    if (HashType == cuckoo_probe) hash_table2 = new cell<T1,T2>[TableSize];
};
    
template <typename T1, typename T2>
Hash_Table<T1,T2>::~Hash_Table(){
    delete[] hash_table1;
    if (type_hash == cuckoo_probe) delete[] hash_table2;
};

//Drucke die Zeile einer Hashtabelle
template <typename T1, typename T2>
std::string Hash_Table<T1,T2>::getCell(size_t i, int j){
    std::string string;
    
    if (j == 0){
        if (i < (table_size)){
            if (hash_table1[i].key!= BLANK){
                string.append(std::to_string(hash_table1[i].key));
                string.append("  ");
                string.append(std::to_string(hash_table1[i].value));
            }else{
                string.append("Leer      Leer");
            } 
        }else{
            string.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
        }

    }else{
        if (hash_table2[i].key!= BLANK){
            string.append(std::to_string(hash_table2[i].key));
            string.append("  ");
            string.append(std::to_string(hash_table2[i].value));
        }else{
            string.append("Leer      Leer");
        } 
    }

    return string;
};

//Gebe die Anzahl der Zellen in der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hash_Table<T1,T2>::getNumCell(){
    size_t sumCell = 0;
    for (size_t i=0; i<table_size; i++) if(hash_table1[i].key!=BLANK) ++sumCell;

    if (type_hash == cuckoo_probe){
        for (size_t j=0; j<table_size; j++) if(hash_table2[j].key!=BLANK) ++sumCell;
        return sumCell;
    }else{
        return sumCell;
    }
};

//Gebe die Größe der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hash_Table<T1,T2>::getTableSize(){
    if (type_hash == cuckoo_probe){
        return table_size*2;
    }else{
        return table_size;
    }
};

//Gebe die Hashtabelle zurück
template <typename T1, typename T2>
cell<T1,T2> * Hash_Table<T1,T2>::getTable(int i){
    if (i == 0){
        return hash_table1;
    }else{
        return hash_table2;
    }
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T1, typename T2>
hash_type Hash_Table<T1,T2>::getHashType(){
    return type_hash;
};

//Gebe eine der zwei Hashfunktionen in der Hashtabelle zurück 
template <typename T1, typename T2>
hash_function Hash_Table<T1,T2>::getHashFunction(int i){
    if (i==0){
        return function1;
    }else{
        return function2;
    }
};

//Gebe eine Zeitmessung für eine Operation in der Hashtabelle zurück
template <typename T1, typename T2>
Benchmark Hash_Table<T1,T2>::getBenchmark(operation_type type){
    if (type == insert_hash_table){
        return benchmark_hash_table[0];
    }else if(type == search_hash_table){
        return benchmark_hash_table[1];
    }else{
        return benchmark_hash_table[2];
    }
};

//Gebe die Zeitmessungen für alle Operationen in der Hashtabelle zurück
template <typename T1, typename T2>
Benchmark * Hash_Table<T1,T2>::getBenchmarkList(){
    return benchmark_hash_table;
};

//Drucke die Hashtabelle
template <typename T1, typename T2>
void Hash_Table<T1,T2>::print(){
    if (type_hash == cuckoo_probe){
        std::cout << "1. Hashtabelle " << std::endl;
        std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "  " << getCell(i,0) << std::endl;  
        std::cout << std::endl;
        
        std::cout << "2. Hashtabelle " << std::endl;
        std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "  " << getCell(i,1) << std::endl; 

    }else{
        std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "  " << getCell(i,0) << std::endl;  
    }
};

//Fuege der Hashtabelle ein Datenelement in der Hashtabelle hinzu
template <typename T1, typename T2>
void Hash_Table<T1,T2>::insert(T1 key, T2 value){
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        T1 prev;

        i = getHash<T1>(key,table_size,function1);
        prev = swapHash<T1>(hash_table1[i].key, BLANK, key);

        if (prev == BLANK || prev == key){
            hash_table1[i].key = key;
            hash_table1[i].value = value;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j<table_size){
            i = (i+j)%table_size;
            prev = swapHash<T1>(hash_table1[i].key, BLANK, key);
            
            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            ++j;
        }

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            prev = swapHash<T1>(hash_table1[i].key, BLANK, key);

            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            ++j;
        }

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            prev = swapHash<T1>(hash_table1[i].key, BLANK, key);

            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            ++j;
        }
    
    //Cuckoo-Hashverfahren
    }else{
        size_t i, j, k;
        size_t max_hash_table;
        T1 prev1, prev2, temp_key;
        T2 temp_value;
        
        i = getHash<T1>(key,table_size,function1);
        j = getHash<T1>(key,table_size,function2);

        k = 1;
        max_hash_table = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*table_size);

        prev1 = swapHash<T1>(hash_table1[i].key, BLANK, key);
    
        if (prev1 == BLANK || prev1 == key){
            hash_table1[i].key = key;
            hash_table1[i].value = value;
            return;
        }

        prev2 = swapHash<T1>(hash_table2[j].key, BLANK, key);
        
        if (prev2 == BLANK || prev2 == key){
            hash_table2[j].key = key;
            hash_table2[j].value = value;
            return;
        }

        while (k<max_hash_table){
            i = (i + k) % (2*table_size);
            j = (j + k) % (2*table_size);

            //Vertausche zwei Schlüssel innerhalb der 1. Hashtabelle
            swapCells<T1,T2>(key,value,i,hash_table1);

            prev1 = swapHash<T1>(hash_table1[i].key, BLANK, key);
            
            if (prev1 == BLANK || prev1 == key){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }

            //Vertausche zwei Schlüssel innerhalb der 2. Hashtabelle
            swapCells<T1,T2>(key,value,j,hash_table2);

            prev2 = swapHash<T1>(hash_table2[j].key, BLANK, key);

            if (prev2 == BLANK || prev2 == key){
                hash_table2[j].key = key;
                hash_table2[j].value = value;
                break;
            }

            ++k;
        }
    }
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T1, typename T2>
void Hash_Table<T1,T2>::insert_List(T1 * keyList, T2 * valueList, size_t cellSize){
    if(cellSize > table_size){
        std::cout << "Die Größe der der Hashtabelle hinzufügenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        cell<T1,T2> * cells;
        cell<T1,T2> * cells_device;
        cell<T1,T2> * hash_table_device1;
        cell<T1,T2> * hash_table_device2;

        std::vector<cell<T1,T2>> cells_vector;

        float duration_upload, duration_run, duration_download, duration_total;
        int min_grid_size, grid_size, block_size;

        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Insert;

        cells_vector.reserve(cellSize);
        
        for (size_t i = 0; i < cellSize ; i++)
            cells_vector.push_back(cell<T1,T2>{keyList[i],valueList[i]});

        cells = cells_vector.data();

        //Ohne Kollisionsauflösung
        if (type_hash == no_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, insert_normal_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&cells_device, &hash_table_device1, &table_size, &function1};

            run.GPUstart();
            cudaLaunchKernel((void*)insert_normal_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Lineare Hashverfahren
        }else if(type_hash == linear_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, insert_linear_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&cells_device, &hash_table_device1, &table_size,&function1};

            run.GPUstart();
            cudaLaunchKernel((void*)insert_linear_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Quadratische Hashverfahren
        }else if(type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, insert_quadratic_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&cells_device, &hash_table_device1, &table_size,&function1};

            run.GPUstart();
            cudaLaunchKernel((void*)insert_quadratic_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, insert_double_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[5] = {&cells_device, &hash_table_device1, &table_size,&function1,&function2};

            run.GPUstart();
            cudaLaunchKernel((void*)insert_double_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, insert_cuckoo_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[6] = {&cells_device, &hash_table_device1, &hash_table_device2, &table_size,&function1,&function2};

            run.GPUstart();
            cudaLaunchKernel((void*)insert_cuckoo_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            cudaMemcpyAsync(hash_table2, hash_table_device2, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();

        Benchmark_Insert.record(insert_hash_table,duration_upload,duration_run,duration_download,duration_total);
        benchmark_hash_table[0] = Benchmark_Insert;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(cells_device);
        cudaFree(cells);
    }
};

//Suche nach einem Schlüssel in der Hashtabelle
template <typename T1, typename T2>
bool Hash_Table<T1,T2>::search(T1 key){     
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        i = getHash<T1>(key,table_size,function1);
        
        if (hash_table1[i].key == key){
            return true;
        }else{
            return false;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j<table_size){
            i = (i+j)%table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Cuckoo-Hashverfahren    
    }else{
        size_t i = getHash<T1>(key,table_size,function1);
        size_t j = getHash<T1>(key,table_size,function2);
        size_t k = 1;

        if (hash_table1[i].key == key) return true;
        if (hash_table2[j].key == key) return true;

        while (k<table_size){
            i = (i + k) % (2*table_size);
            j = (j + k) % (2*table_size);
            
            if (hash_table1[i].key == key) return true;
            if (hash_table2[j].key == key) return true;

            ++k;
        }
        return false;
    }
};

//Suche nach einem Array von Schlüsseln in der Hashtabelle gleichzeitig.
template <typename T1, typename T2>
void Hash_Table<T1,T2>::search_List(T1 * keyList, size_t cellSize){
    if(cellSize > table_size){
        std::cout << "Die Größe der nach suchenden Schlüsseln in der Hashtabelle muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        T1 * keyListResult = new T1[cellSize];

        T1 * keyList_device;
        T1 * keyListResult_device;
        cell<T1,T2> * hash_table_device1;
        cell<T1,T2> * hash_table_device2;
        
        float duration_upload, duration_run, duration_download, duration_total;
        int min_grid_size, grid_size, block_size;
        size_t sum_found;
        
        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Search;

        //Ohne Kollisionsauflösung
        if (type_hash == no_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T1)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, search_normal_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[5] = {&keyList_device, &keyListResult_device, &hash_table_device1, &table_size,&function1};

            run.GPUstart();
            cudaLaunchKernel((void*)search_normal_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T1)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Lineare Hashverfahren
        }else if(type_hash == linear_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, search_linear_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[5] = {&keyList_device, &keyListResult_device, &hash_table_device1, &table_size,&function1};

            run.GPUstart();
            cudaLaunchKernel((void*)search_linear_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T1)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Quadratische Hashverfahren
        }else if(type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, search_quadratic_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[5] = {&keyList_device, &keyListResult_device, &hash_table_device1, &table_size,&function1};

            run.GPUstart();
            cudaLaunchKernel((void*)search_quadratic_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T1)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, search_double_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[6] = {&keyList_device, &keyListResult_device, &hash_table_device1, &table_size,&function1,&function2};

            run.GPUstart();
            cudaLaunchKernel((void*)search_double_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T1)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

             total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T1)*cellSize);
            
            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, search_cuckoo_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[7] = {&keyList_device, &keyListResult_device, &hash_table_device1, &hash_table_device2, &table_size,&function1,&function2};

            run.GPUstart();
            cudaLaunchKernel((void*)search_cuckoo_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T1)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        sum_found = 0;
        for (size_t i = 0; i<cellSize; i++) if (keyListResult[i] == BLANK) ++sum_found; 

        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();

        Benchmark_Search.record(search_hash_table,duration_upload,duration_run,duration_download,duration_total);
        benchmark_hash_table[1] = Benchmark_Search;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(keyList_device);
        cudaFree(keyListResult_device);
        cudaFree(keyListResult);
    }
};

//Lösche Schlüssel aus der Hashtabelle
template <typename T1, typename T2>
void Hash_Table<T1,T2>::deleteKey(T1 key){
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        T1 prev;

        i = getHash<T1>(key,table_size,function1);
        prev = swapHash<T1>(hash_table1[i].key, key, BLANK);

        if (prev == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].value = BLANK;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j<table_size){
            i = (i+j)%table_size;
            prev = swapHash<T1>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].value = BLANK;
                break;
            }
            ++j;
        }

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            prev = swapHash<T1>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].value = BLANK;
                break;
            }
            ++j;
        }

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        T1 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2)<table_size){
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            prev = swapHash<T1>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].value = BLANK;
                break;
            }
            ++j;
        }

    //Cuckoo-Hashverfahren    
    }else{
        size_t i, j, k;
        T1 prev1, prev2;

        i = getHash<T1>(key,table_size,function1);
        j = getHash<T1>(key,table_size,function2);
        k = 1;

        prev1 = swapHash<T1>(hash_table1[i].key, key, BLANK);

        if (prev1 == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].value = BLANK;
            return;
        }

        prev2 = swapHash<T1>(hash_table2[j].key, key, BLANK);

        if (prev2 == BLANK){
            hash_table2[j].key = BLANK;
            hash_table2[j].value = BLANK;
            return;
        }

        while (k<table_size){
            i = (i + k) % (2*table_size);
            j = (j + k) % (2*table_size);

            prev1 = swapHash<T1>(hash_table1[i].key, key, BLANK);

            if (prev1 == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].value = BLANK;
                break;
            }

            prev2 = swapHash<T1>(hash_table2[j].key, key, BLANK);

            if (prev2 == BLANK){
                hash_table2[j].key = BLANK;
                hash_table2[j].value = BLANK;
                break;
            }
            ++k;
        }
    }
};

//Lösche eine Liste von Schlüssel von der Hashtabelle    
template <typename T1, typename T2>
void Hash_Table<T1,T2>::delete_List(T1 * keyList, size_t cellSize){
    if(cellSize > table_size){
        std::cout << "Die Größe der zu löschenden Schlüsseln in der Hashtabelle muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        T1 * keyList_device;
        cell<T1,T2> * hash_table_device1;
        cell<T1,T2> * hash_table_device2;
        
        float duration_upload, duration_run, duration_download, duration_total;
        int min_grid_size, grid_size, block_size;
        
        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Delete;

        //Ohne Kollisionsauflösung
        if (type_hash == no_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, delete_normal_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&keyList_device, &hash_table_device1, &table_size, &function1};

            run.GPUstart();
            cudaLaunchKernel((void*)delete_normal_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Lineare Hashverfahren
        }else if(type_hash == linear_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, delete_linear_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&keyList_device, &hash_table_device1, &table_size, &function1};

            run.GPUstart();
            cudaLaunchKernel((void*)delete_linear_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Quadratische Hashverfahren
        }else if(type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, delete_quadratic_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[4] = {&keyList_device, &hash_table_device1, &table_size, &function1};

            run.GPUstart();
            cudaLaunchKernel((void*)delete_quadratic_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, delete_double_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[5] = {&keyList_device, &hash_table_device1, &table_size, &function1, &function2};

            run.GPUstart();
            cudaLaunchKernel((void*)delete_double_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(T1)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,keyList,sizeof(T1)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, delete_cuckoo_kernel<T1,T2>, 0, 0);
            grid_size = ((size_t)(cellSize)+block_size-1)/block_size;
            dim3 block(block_size);
            dim3 grid(grid_size);
            
            void *args[6] = {&keyList_device, &hash_table_device1, &hash_table_device2, &table_size, &function1, &function2};

            run.GPUstart();
            cudaLaunchKernel((void*)delete_cuckoo_kernel<T1,T2>,grid,block,args,0,run.getStream());
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            cudaMemcpyAsync(hash_table2, hash_table_device2, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();

        Benchmark_Delete.record(delete_hash_table,duration_upload,duration_run,duration_download,duration_total);
        benchmark_hash_table[2] = Benchmark_Delete;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(keyList_device);
    }
};

#endif