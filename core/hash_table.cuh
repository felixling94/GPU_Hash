#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../core/hash_methods.cuh>
#include <../tools/timer.cuh>
#include <../tools/benchmark.h>

template <typename T>
Hash_Table<T>::Hash_Table():type_hash(no_probe),function1(modulo),function2(modulo),table_size(2){
    hash_table1 = new cell<T>[2];
};

template <typename T>
Hash_Table<T>::Hash_Table(hash_type HashType, hash_function function_1, hash_function function_2, size_t TableSize):
type_hash(HashType),function1(function_1),function2(function_2),table_size(TableSize){
    hash_table1 = new cell<T>[TableSize];
    if (HashType == cuckoo_probe) hash_table2 = new cell<T>[TableSize];
};
    
template <typename T>
Hash_Table<T>::~Hash_Table(){
    delete[] hash_table1;
    if (type_hash == cuckoo_probe) delete[] hash_table2;
};

//Drucke die Zeile einer Hashtabelle
template <typename T>
std::string Hash_Table<T>::getCell(size_t i, int j){
    std::string string;
    
    if (j == 0){
        if (i < (table_size)){
            if (hash_table1[i].key!= BLANK){
                string.append(std::to_string(hash_table1[i].key));
                string.append(",");
                string.append(std::to_string(hash_table1[i].key_length));
            }else{
                string.append("Leer");
                string.append(",");
                string.append("Leer");
            } 
        }else{
            string.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
        }

    }else{
        if (hash_table2[i].key!= BLANK){
            string.append(std::to_string(hash_table2[i].key));
            string.append(",");
            string.append(std::to_string(hash_table2[i].key_length));
        }else{
            string.append("Leer");
            string.append(",");
            string.append("Leer");
        } 
    }

    return string;
};

//Gebe die Anzahl der Zellen in der Hashtabelle zurück
template <typename T>
size_t Hash_Table<T>::getNumCell(){
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
template <typename T>
size_t Hash_Table<T>::getTableSize(){
    if (type_hash == cuckoo_probe){
        return table_size*2;
    }else{
        return table_size;
    }
};

//Gebe die Hashtabelle zurück
template <typename T>
cell<T> * Hash_Table<T>::getTable(int i){
    if (i == 0){
        return hash_table1;
    }else{
        return hash_table2;
    }
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T>
hash_type Hash_Table<T>::getHashType(){
    return type_hash;
};

//Gebe eine der zwei Hashfunktionen in der Hashtabelle zurück 
template <typename T>
hash_function Hash_Table<T>::getHashFunction(int i){
    if (i==0){
        return function1;
    }else{
        return function2;
    }
};

//Gebe eine Zeitmessung für eine Operation in der Hashtabelle zurück
template <typename T>
Benchmark Hash_Table<T>::getBenchmark(operation_type type){
    if (type == insert_hash_table){
        return benchmark_hash_table[0];
    }else if(type == search_hash_table){
        return benchmark_hash_table[1];
    }else{
        return benchmark_hash_table[2];
    }
};

//Gebe die Zeitmessungen für alle Operationen in der Hashtabelle zurück
template <typename T>
Benchmark * Hash_Table<T>::getBenchmarkList(){
    return benchmark_hash_table;
};

//Drucke die Hashtabelle
template <typename T>
void Hash_Table<T>::print(){
    if (type_hash == cuckoo_probe){
        std::cout << "1. Hashtabelle " << std::endl;
        std::cout << "Index" << "," << "Schlüsselcode" << "," << "Länge" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,0) << std::endl;  
        std::cout << std::endl;
        
        std::cout << "2. Hashtabelle " << std::endl;
        std::cout << "Index" << "," << "Schlüsselcode" << "," << "Länge" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,1) << std::endl; 

    }else{
        std::cout << "Index" << "," << "Schlüsselcode" << "," << "Länge" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,0) << std::endl;  
    }
};

//Fuege der Hashtabelle ein Datenelement in der Hashtabelle hinzu
template <typename T>
void Hash_Table<T>::insert(T key, T key_length){
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        T prev;

        i = getHash<T>(key_length,table_size,function1);
        prev = swapHash<T>(hash_table1[i].key, BLANK, key);

        if (prev == BLANK || prev == key){
            hash_table1[i].key = key;
            hash_table1[i].key_length = key_length;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+j)%table_size;
            prev = swapHash<T>(hash_table1[i].key, BLANK, key);
            
            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].key_length = key_length;
                break;
            }
            ++j;
        }

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while((j/2) < table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            prev = swapHash<T>(hash_table1[i].key, BLANK, key);

            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].key_length = key_length;
                break;
            }
            ++j;
        }

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+getHashProbe<T>(key_length,j,table_size,function2))%table_size;
            prev = swapHash<T>(hash_table1[i].key, BLANK, key);

            if (prev == BLANK || prev == key){
                hash_table1[i].key = key;
                hash_table1[i].key_length = key_length;
                break;
            }
            ++j;
        }
    
    //Cuckoo-Hashverfahren
    }else{
        size_t i, j, k;
        size_t max_hash_table;
        T prev1, prev2;
        
        i = getHash<T>(key_length,table_size,function1);
        j = getHash<T>(key_length,table_size,function2);

        k = 1;
        max_hash_table = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*table_size);

        prev1 = swapHash<T>(hash_table1[i].key, BLANK, key);
    
        if (prev1 == BLANK || prev1 == key){
            hash_table1[i].key = key;
            hash_table1[i].key_length = key_length;
            return;
        }

        prev2 = swapHash<T>(hash_table2[j].key, BLANK, key);
        
        if (prev2 == BLANK || prev2 == key){
            hash_table2[j].key = key;
            hash_table2[j].key_length = key_length;
            return;
        }

        while (k < max_hash_table){
            i = (i + k) % (2*table_size);
            j = (j + k) % (2*table_size);

            //Vertausche zwei Schlüssel innerhalb der 1. Hashtabelle
            swapCells<T>(key,key_length,i,hash_table1);

            prev1 = swapHash<T>(hash_table1[i].key, BLANK, key);
            
            if (prev1 == BLANK || prev1 == key){
                hash_table1[i].key = key;
                hash_table1[i].key_length = key_length;
                break;
            }

            //Vertausche zwei Schlüssel innerhalb der 2. Hashtabelle
            swapCells<T>(key,key_length,j,hash_table2);

            prev2 = swapHash<T>(hash_table2[j].key, BLANK, key);

            if (prev2 == BLANK || prev2 == key){
                hash_table2[j].key = key;
                hash_table2[j].key_length = key_length;
                break;
            }

            ++k;
        }
    }
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T>
void Hash_Table<T>::insert_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock){
    size_t cellSize{(size_t)(numBlocks*numThreadsProBlock)};

    if(cellSize > table_size){
        std::cout << "Die Größe der der Hashtabelle hinzufügenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        cell<T> * cells;
        cell<T> * cells_device;
        cell<T> * hash_table_device1;
        cell<T> * hash_table_device2;

        std::vector<cell<T>> cells_vector;

        float duration_upload, duration_run, duration_download, duration_total;
        size_t num_insert;

        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Insert;

        cells_vector.reserve(cellSize);
        
        for (size_t i = 0; i < cellSize ; i++)
            cells_vector.push_back(cell<T>{keyList[i],keyLengthList[i]});

        cells = cells_vector.data();
 
        dim3 num_Blocks(numBlocks);
        dim3 num_ThreadsPerBlock(numThreadsProBlock);

        //Ohne Kollisionsauflösung, mit linearen Hashverfahren, mit quadratischen Hashverfahren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            run.GPUstart();

            if (type_hash == linear_probe){
                insert_linear<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                insert_quadratic<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }else{
                insert_normal<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }

            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            run.GPUstart();
            insert_double<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1, function2);
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            run.GPUstart();
            insert_cuckoo<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            cudaMemcpyAsync(hash_table2, hash_table_device2, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();
        num_insert = getNumCell();

        Benchmark_Insert.record(insert_hash_table,duration_upload,duration_run,duration_download,duration_total,num_insert,type_hash);
        benchmark_hash_table[0] = Benchmark_Insert;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(cells_device);
        cudaFree(cells);
    }
};

//Suche nach einem Schlüssel in der Hashtabelle
template <typename T>
bool Hash_Table<T>::search(T key, T key_length){   
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        i = getHash<T>(key_length,table_size,function1);
        
        if (hash_table1[i].key == key){
            return true;
        }else{
            return false;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+j)%table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while((j/2) < table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+getHashProbe<T>(key_length,j,table_size,function2))%table_size;
            if (hash_table1[i].key == key) return true;
            ++j;
        }
        return false;

    //Cuckoo-Hashverfahren    
    }else{
        size_t i = getHash<T>(key_length,table_size,function1);
        size_t j = getHash<T>(key_length,table_size,function2);
        size_t k = 1;

        if (hash_table1[i].key == key) return true;
        if (hash_table2[j].key == key) return true;

        while (k < table_size){
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
template <typename T>
void Hash_Table<T>::search_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock){
    size_t cellSize{(size_t)(numBlocks*numThreadsProBlock)};

    if(cellSize > table_size){
        std::cout << "Die Größe der nach suchenden Schlüsseln in der Hashtabelle muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        cell<T> * cells;
        T * keyListResult = new T[cellSize];
        T * keyListResult_device;
        cell<T> * keyList_device;
        cell<T> * hash_table_device1;
        cell<T> * hash_table_device2;

        std::vector<cell<T>> cells_vector;
        
        float duration_upload, duration_run, duration_download, duration_total;
        size_t sum_found;
        
        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Search;

        cells_vector.reserve(cellSize);
        
        for (size_t i = 0; i < cellSize ; i++)
            cells_vector.push_back(cell<T>{keyList[i],keyLengthList[i]});

        cells = cells_vector.data();

        dim3 num_Blocks(numBlocks);
        dim3 num_ThreadsPerBlock(numThreadsProBlock);

        //Ohne Kollisionsauflösung, mit linearem und quadratischem Sondieren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();

            if (type_hash == linear_probe){
                search_linear<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                search_quadratic<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }else{
                search_normal<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }

            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            search_double<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1, function2);
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T)*cellSize);
            
            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            search_cuckoo<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        sum_found = 0;
        for (size_t i = 0; i<cellSize; i++) if (keyListResult[i] == BLANK) ++sum_found; 

        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();

        Benchmark_Search.record(search_hash_table,duration_upload,duration_run,duration_download,duration_total,sum_found,type_hash);
        benchmark_hash_table[1] = Benchmark_Search;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(keyList_device);
        cudaFree(keyListResult_device);
        cudaFree(keyListResult);
    }
};

//Lösche Schlüssel aus der Hashtabelle
template <typename T>
void Hash_Table<T>::deleteKey(T key, T key_length){
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        T prev;

        i = getHash<T>(key_length,table_size,function1);
        prev = swapHash<T>(hash_table1[i].key, key, BLANK);

        if (prev == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].key_length = BLANK;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+j)%table_size;
            prev = swapHash<T>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].key_length = BLANK;
                break;
            }
            ++j;
        }

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while((j/2) < table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            prev = swapHash<T>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].key_length = BLANK;
                break;
            }
            ++j;
        }

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        T prev;
        
        i = getHash<T>(key_length,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+getHashProbe<T>(key_length,j,table_size,function2))%table_size;
            prev = swapHash<T>(hash_table1[i].key, key, BLANK);

            if (prev == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].key_length = BLANK;
                break;
            }
            ++j;
        }

    //Cuckoo-Hashverfahren    
    }else{
        size_t i, j, k;
        T prev1, prev2;

        i = getHash<T>(key_length,table_size,function1);
        j = getHash<T>(key_length,table_size,function2);
        k = 1;

        prev1 = swapHash<T>(hash_table1[i].key, key, BLANK);

        if (prev1 == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].key_length = BLANK;
            return;
        }

        prev2 = swapHash<T>(hash_table2[j].key, key, BLANK);

        if (prev2 == BLANK){
            hash_table2[j].key = BLANK;
            hash_table2[j].key_length = BLANK;
            return;
        }

        while (k < table_size){
            i = (i + k) % (2*table_size);
            j = (j + k) % (2*table_size);

            prev1 = swapHash<T>(hash_table1[i].key, key, BLANK);

            if (prev1 == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].key_length = BLANK;
                break;
            }

            prev2 = swapHash<T>(hash_table2[j].key, key, BLANK);

            if (prev2 == BLANK){
                hash_table2[j].key = BLANK;
                hash_table2[j].key_length = BLANK;
                break;
            }
            ++k;
        }
    }
};
 
//Lösche eine Liste von Schlüssel von der Hashtabelle    
template <typename T>
void Hash_Table<T>::delete_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock){
    size_t cellSize{(size_t)(numBlocks*numThreadsProBlock)};

    if(cellSize > table_size){
        std::cout << "Die Größe der zu löschenden Schlüsseln in der Hashtabelle muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        cell<T> * cells;
        cell<T> * keyList_device;
        cell<T> * hash_table_device1;
        cell<T> * hash_table_device2;

        std::vector<cell<T>> cells_vector;
        
        float duration_upload, duration_run, duration_download, duration_total;
        size_t num_cells_prev, num_cells_deleted;
        
        GPUTimer upload, run, download, total;
        
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        Benchmark Benchmark_Delete;

        num_cells_prev = getNumCell();

        cells_vector.reserve(cellSize);
        
        for (size_t i = 0; i < cellSize ; i++)
            cells_vector.push_back(cell<T>{keyList[i],keyLengthList[i]});

        cells = cells_vector.data();
        
        dim3 num_Blocks(numBlocks);
        dim3 num_ThreadsPerBlock(numThreadsProBlock);

        //Ohne Kollisionsauflösung, mit linearem und quadratischem Sondieren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();

            if (type_hash == linear_probe){
                delete_linear<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                delete_quadratic<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }else{
                delete_normal<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }

            run.GPUstop();
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            delete_double<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1, function2);
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            delete_cuckoo<T><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(keyList_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            cudaMemcpyAsync(hash_table2, hash_table_device2, sizeof(cell<T>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();
        }
        
        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();

        num_cells_deleted = num_cells_prev - getNumCell();

        Benchmark_Delete.record(delete_hash_table,duration_upload,duration_run,duration_download,duration_total,num_cells_deleted,type_hash);
        benchmark_hash_table[2] = Benchmark_Delete;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(keyList_device);
    }
};

#endif