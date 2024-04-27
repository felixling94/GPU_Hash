#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../core/hash_methods.cuh>
#include <../tools/timer.cuh>
#include <../tools/benchmark.h>

//Voreingestellter Konstruktor für die Erstellung einer einfachen Hashtabelle
template <typename T1, typename T2>
Hash_Table<T1,T2>::Hash_Table():type_hash(no_probe),function1(modulo),function2(modulo),table_size(2){
    hash_table1 = new cell<T1,T2>[2];
};

//Konstruktor für die Erstellung einer Hashtabelle
template <typename T1, typename T2>
Hash_Table<T1,T2>::Hash_Table(hash_type HashType, hash_function function_1, hash_function function_2, size_t TableSize,
                          int NumBlocks, int NumThreadsPerBlock):
type_hash(HashType),function1(function_1),function2(function_2),table_size(TableSize){
    hash_table1 = new cell<T1,T2>[TableSize];
    if (HashType == cuckoo_probe) hash_table2 = new cell<T1,T2>[TableSize];

    if (NumBlocks > 0 && NumThreadsPerBlock > 0){
        dimension_kernel.num_blocks = NumBlocks;
        dimension_kernel.num_threads_per_block = NumThreadsPerBlock;
    }
};

//Destruktor einer Hashtabelle    
template <typename T1, typename T2>
Hash_Table<T1,T2>::~Hash_Table(){
    delete[] hash_table1;
    delete[] benchmark_hash_table;
    if (type_hash == cuckoo_probe) delete[] hash_table2;
};

//Drucke die Zeile einer Hashtabelle
template <typename T1, typename T2>
std::string Hash_Table<T1,T2>::getCell(size_t i, int j){
    std::string string;
    
    if (j == 0){
        if (i < (table_size)){
            if (hash_table1[i].value!= BLANK){
                string.append(std::to_string(hash_table1[i].key));
                string.append(",");
                string.append(std::to_string(hash_table1[i].value));
            }else{
                string.append("Leer");
                string.append(",");
                string.append("Leer");
            } 
        }else{
            string.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
        }
    }else{
        if (hash_table2[i].value!= BLANK){
            string.append(std::to_string(hash_table2[i].key));
            string.append(",");
            string.append(std::to_string(hash_table2[i].value));
        }else{
            string.append("Leer");
            string.append(",");
            string.append("Leer");
        } 
    }
    return string;
};

//Gebe die Anzahl der gespeicherten Zellen in der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hash_Table<T1,T2>::getNumCell(){
    size_t sumCell = 0;
    for (size_t i=0; i<table_size; i++) if(hash_table1[i].value!=BLANK) ++sumCell;

    if (type_hash == cuckoo_probe){
        for (size_t j=0; j<table_size; j++) if(hash_table2[j].value!=BLANK) ++sumCell;
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

//Gebe eine von zwei Hashtabellen zurück
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

//Gebe die Anzahl von Blöcken und Threads in einem Block zurück pro:
//a. Speicherung von Schlüsseln
//b. Suche nach Schlüsseln
//c. Löschung von Schlüsseln 
template <typename T1, typename T2>
kernel_dimension Hash_Table<T1,T2>::getKernelDimension(){
    return dimension_kernel;
}

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
        std::cout << "Index" << "," << "Schlüssel" << "," << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,0) << std::endl;  
        std::cout << std::endl;
        
        std::cout << "2. Hashtabelle " << std::endl;
        std::cout << "Index" << "," << "Schlüssel" << "," << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,1) << std::endl; 
    }else{
        std::cout << "Index" << "," << "Schlüssel" << "," << "Wert" << std::endl;
        for(size_t i = 0; i < table_size; i++) std::cout << i << "," << getCell(i,0) << std::endl;  
    }
};

/* Füge der Hashtabelle ein Paar von einem Schlüssel und dessen Wert durch 
    ausgewählte offene Hashverfahren hinzu:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
void Hash_Table<T1,T2>::insert(T1 key, T2 value){        
    //1. Kollisionsauflösung mit ausgewählten Hashverfahren: 
    //1a. ohne Kollisionsauflösung, linearem Sondieren, quadratischem Sondieren oder
    //1b. doppelten Hashverfahren oder
    //1c. Cuckoo-Hashverfahren
    if (type_hash == no_probe){
        //2a. Deklariere die Variablen
        size_t i;
        T2 prev;

        //2b. Setze den Hashwert eines Schlüssels
        i = getHash<T1>(key,table_size,function1);
        //2c. Vertausche einen Schlüssel mit dem anderen in der Hashtabelle
        prev = swapHash<T2>(hash_table1[i].value, BLANK, value);

        //2d1. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
        //2d2. Belege die Zelle in der Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
        if (prev == BLANK || prev == value){
            hash_table1[i].key = key;
            hash_table1[i].value = value;
        }
    }else if(type_hash == linear_probe){
        //2a. Deklariere die Variablen
        size_t i, j;
        T2 prev;
        
        //2b. Setz den Hashwert eines Schlüssels
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        //2c. Schleifendurchlauf durch die Größe einer Hashtabelle
        while(j < table_size){
            //2c1. Berechne den neuen Hashwert eines Schlüssels durch lineare Sondierungsfunktion
            i = (i+j)%table_size;
            //2c2. Vertausche einen Schlüssel mit dem anderen in der Hashtabelle
            prev = swapHash<T2>(hash_table1[i].value, BLANK, value);
            
            //2c3a. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
            //2c3b. Belege die Zelle in der Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
            if (prev == BLANK || prev == value){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            //2c4. Erhöhe den Hashwert eines Schlüssels
            ++j;
        }
    }else if(type_hash == quadratic_probe){
        //2a. Deklariere die Variablen
        size_t i, j;
        T2 prev;
        
        //2b. Setze den Hashwert eines Schlüssels
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        //2c. Führe einen Schleifendurchlauf durch die doppelte Größe einer Hashtabelle aus
        while((j/2) < table_size){
            //2c1. Berechne den neuen Hashwert eines Schlüssels durch quadratische Sondierungsfunktion
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            //2c2. Vertausche einen Schlüssel mit dem anderen in der Hashtabelle
            prev = swapHash<T2>(hash_table1[i].value, BLANK, value);

            //2c3a. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
            //2c3b. Belege die Zelle in der Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
            if (prev == BLANK || prev == value){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            //2c4. Erhöhe den Hashwert eines Schlüssels
            ++j;
        }
    }else if (type_hash == double_probe){
        //2a. Deklariere die Variablen
        size_t i, j;
        T2 prev;

        //2b. Setze den Hashwert eines Schlüssels
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        //2c. Führe einen Schleifendurchlauf durch die Größe einer Hashtabelle aus
        while(j < table_size){
            //2c1. Berechne den neuen Hashwert eines Schlüssels durch eine andere Hashfunktion
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            //2c2. Vertausche einen Schlüssel mit dem anderen in der Hashtabelle
            prev = swapHash<T2>(hash_table1[i].value, BLANK, value);

            //2c3a. Überprüfe, ob die Zelle in der Hashtabelle belegt ist
            //2c3b. Belege die Zelle in der Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
            if (prev == BLANK || prev == value){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }
            //2c4. Erhöhe den Hashwert eines Schlüssels
            ++j;
        }
    }else{
        //2a. Deklariere die Variablen
        size_t i, j, k, max_hash_table;
        T2 prev1, prev2;
        
        //2b. Setze die Hashwerte eines Schlüssels
        i = getHash<T1>(key,table_size,function1);
        j = getHash<T1>(key,table_size,function2);
        k = 1;
        max_hash_table = (size_t)(((int)(100+LOOP_PERCENTAGE))/100*table_size);
 
        //2c. Vertausche einen Schlüssel mit dem anderen in der ersten Hashtabelle
        prev1 = swapHash<T2>(hash_table1[i].value, BLANK, value);
    
        //2d1. Überprüfe, ob die Zelle in der ersten Hashtabelle belegt ist
        //2d2. Belege die Zelle in der ersten Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
        if (prev1 == BLANK || prev1 == value){
            hash_table1[i].key = key;
            hash_table1[i].value = value;
            return;
        }
        //2e. Vertausche einen Schlüssel mit dem anderen in der zweiten Hashtabelle
        prev2 = swapHash<T1>(hash_table2[j].value, BLANK, value);
        
        //2f1. Überprüfe, ob die Zelle in der zweiten Hashtabelle belegt ist
        //2f2. Belege die Zelle in der zweiten Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
        if (prev2 == BLANK || prev2 == value){
            hash_table2[j].key = key;
            hash_table2[j].value = value;
            return;
        }
        //2g. Führe einen Schleifendurchlauf durch eine bestimmte Anzahl an Schleifen aus
        while (k < max_hash_table){
            //2g1. Berechne die neuen Hashwerte eines Schlüssels durch zwei lineare Sondierungsfunktionen
            i = (i + k) % table_size;
            j = (j + k) % table_size;

            //2g2. Vertausche einen Schlüssel mit dem anderen in der ersten Hashtabelle
            swapCells<T1,T2>(key,value,i,hash_table1);
            prev1 = swapHash<T2>(hash_table1[i].value, BLANK, value);
            
            //2g3a. Überprüfe, ob die Zelle in der ersten Hashtabelle belegt ist
            //2g3b. Belege die Zelle in der ersten Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
            if (prev1 == BLANK || prev1 == value){
                hash_table1[i].key = key;
                hash_table1[i].value = value;
                break;
            }

            //2g4. Vertausche einen Schlüssel mit dem anderen in der zweiten Hashtabelle
            swapCells<T1,T2>(key,value,j,hash_table2);
            prev2 = swapHash<T2>(hash_table2[j].value, BLANK, value);
        
            //2g5a. Überprüfe, ob die Zelle in der zweiten Hashtabelle belegt ist
            //2g5b. Belege die Zelle in der zweiten Hashtabelle mit den Werten vom Schlüssel und dessen Länge bei freiem Speicherplatz
            if (prev2 == BLANK || prev2 == value){
                hash_table2[j].key = key;
                hash_table2[j].value = value;
                break;
            }
            //2g6. Erhöhe den Hashwert eines Schlüssels
            ++k;
        }
    }
};

/* Füge der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig 
    durch ausgewählte offene Hashverfahren hinzu:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
void Hash_Table<T1,T2>::insert_List(T1 * keyList, T2 * valueList, size_t cellSize){
    if((cellSize > table_size) && type_hash != cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle einer Hashtabelle zu speichernden Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe einer Hashtabelle betragen." << std::endl;

    }else if((cellSize > (2*table_size)) && type_hash == cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle von zwei Hashtabellen zu speichernden Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe von zwei Hashtabellen betragen." << std::endl;

    }else{
        //1. Deklariere die Variablen
        cell<T1,T2> * cells, * cells_device, * hash_table_device1, * hash_table_device2;
        std::vector<cell<T1,T2>> cells_vector;

        float duration_upload, duration_run, duration_download, duration_total;
        size_t num_insert;

        GPUTimer upload, run, download, total;
        Benchmark Benchmark_Insert;

        //2. Setze alle Variablen
        duration_upload = 0; 
        duration_run = 0; 
        duration_download = 0;
        duration_total = 0;

        //2a. Erstelle eine Liste von Zellen bestehend aus Schlüsseln und deren Werten
        cells_vector.reserve(cellSize);
        
        for (size_t i = 0; i < cellSize ; i++)
            cells_vector.push_back(cell<T1,T2>{keyList[i],valueList[i]});

        cells = cells_vector.data();
 
        //2b. Setze die Kerneldimension bei falscher Block - und Threadzahl pro Block
        if (dimension_kernel.num_blocks < 1 || dimension_kernel.num_threads_per_block < 1 ||
            cellSize != (size_t)(dimension_kernel.num_blocks*dimension_kernel.num_threads_per_block)){
            dimension_kernel.num_blocks = (int) cellSize;
            dimension_kernel.num_threads_per_block = 1;
        }
        
        //2c. Setze die Kerneldimension und die Speichergröße von "shared memory" für die Kernelausführung von offenen Hashverfahren
        dim3 num_Blocks(dimension_kernel.num_blocks,1,1);
        dim3 num_ThreadsPerBlock(dimension_kernel.num_threads_per_block,1,1);
        size_t shared_memory_size{dimension_kernel.num_threads_per_block*sizeof(cell<T1,T2>)};

        //3. Kerneldurchführung mit ausgewählten Hashverfahren: 
        //3a. ohne Kollisionsauflösung, linearem Sondieren, quadratischem Sondieren oder
        //3b. doppelten Hashverfahren oder
        //3c. Cuckoo-Hashverfahren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //4a. Reserviere Speicherplätze auf der GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            //4b. Beginn der Gesamtzeitmessung
            total.GPUstart();

            //4c1. Beginn der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstart();
            //4c2. Übertrage Daten aus der Hashtabelle und eingegebenen Zellen von CPU auf GPU
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            //4c3. Ende der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstop();

            //4d1. Beginn der Zeitmessung der Kernelausführung von offenen Hashverfahren
            //4d2. Wahl von drei offenen Hashverfahren: ohne Kollisionsauflösung, linearem und quadratischem Sondieren
            run.GPUstart();

            if (type_hash == linear_probe){
                //4d3. Füge der Hashtabelle alle eingegebenen Zellen auf der GPU durch lineares Sondieren hinzu
                //insert_linear<T1,T2><<<num_Blocks,num_ThreadsPerBlock,0,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
                insert_linear<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                //4d3. Füge der Hashtabelle alle eingegebenen Zellen auf der GPU durch quadratisches Sondieren hinzu
                insert_quadratic<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }else{
                //4d3. Füge der Hashtabelle alle eingegebenen Zellen auf der GPU ohne Kollisionsauflösung hinzu
                insert_normal<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1);
            }

            //4d4. Ende der Zeitmessung der Kernelausführung von offenen Hashverfahren
            run.GPUstop(); 

            //4e1. Beginn der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstart();
            //4e2. Übertrage Daten aus der Hashtabelle von GPU auf CPU
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            //4e3. Ende der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstop();

            //4f. Ende der Gesamtzeitmessung
            total.GPUstop();

        }else if (type_hash == double_probe){
            //4a. Reserviere Speicherplätze auf der GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            //4b. Beginn der Gesamtzeitmessung
            total.GPUstart();

            //4c1. Beginn der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstart();
            //4c2. Übertrage Daten aus der Hashtabelle und eingegebenen Zellen von CPU auf GPU
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            //4c3. Ende der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstop();

            //4d1. Beginn der Zeitmessung der Kernelausführung von doppelten Hashverfahren
            run.GPUstart();
            //4d2. Füge der Hashtabelle alle eingegebenen Zellen auf der GPU durch doppelte Hashverfahren hinzu
            insert_double<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(cells_device, hash_table_device1, table_size, function1, function2);
            //4d3. Ende der Zeitmessung der Kernelausführung von doppelten Hashverfahren
            run.GPUstop(); 

            //4e1. Beginn der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstart();
            //4e2. Übertrage Daten aus der Hashtabelle von GPU auf CPU
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            //4e3. Ende der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstop();

            //4f. Ende der Gesamtzeitmessung
            total.GPUstop();

        }else{
            //4a. Reserviere Speicherplätze auf der GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&cells_device,sizeof(cell<T1,T2>)*cellSize);

            //4b. Beginn der Gesamtzeitmessung
            total.GPUstart();
            
            //4c1. Beginn der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstart();
            //4c2. Übertrage Daten aus den Hashtabellen und eingegebenen Zellen von CPU auf GPU
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(cells_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            //4c3. Ende der Zeitmessung vom Hochladen der Daten von CPU auf GPU
            upload.GPUstop();

            //4d1. Beginn der Zeitmessung der Kernelausführung von Cuckoo-Hashverfahren
            run.GPUstart();
            //4d2. Füge der Hashtabelle alle eingegebenen Zellen auf der GPU durch Cuckoo-Hashverfahren hinzu
            insert_cuckoo<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(cells_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
            //4d3. Ende der Zeitmessung der Kernelausführung von Cuckoo-Hashverfahren
            run.GPUstop(); 
            
            //4e1. Beginn der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstart();
            //4e2. Übertrage Daten aus den Hashtabellen von GPU auf CPU
            cudaMemcpyAsync(hash_table1, hash_table_device1, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            cudaMemcpyAsync(hash_table2, hash_table_device2, sizeof(cell<T1,T2>)*table_size, cudaMemcpyDeviceToHost,download.getStream());
            //4e3. Ende der Zeitmessung vom Herunterladen der Daten von GPU auf CPU
            download.GPUstop();

            //4f. Ende der Gesamtzeitmessung
            total.GPUstop();
        }
        
        //5. Führe die Messung von Daten aus von:
        //5a. Dauer vom Hochladen von Daten
        //5b. Dauer von der Kernelausführung
        //5c. Dauer vom Herunterladen von Daten
        //5d. Gesamtdauer der Kernelausführung
        //5e. Anzahl von tatsächlichen gespeicherten Zellen
        duration_upload = upload.getGPUDuration();
        duration_run = run.getGPUDuration();
        duration_download = download.getGPUDuration();
        duration_total = total.getGPUDuration();
        num_insert = getNumCell();

        Benchmark_Insert.record(insert_hash_table,duration_upload,duration_run,duration_download,duration_total,num_insert,type_hash);
        benchmark_hash_table[0] = Benchmark_Insert;

        //6. Gebe die Speicherplätze von den auf CPU und GPU befindlichen Hashtabellen und Zellen frei
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(cells_device);
        cudaFree(cells);
    }
};

/* Suche nach einem Schlüssel und dessen Wert in einer von zwei Hashtabellen durch 
    ausgewählte offene Hashverfahren:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
bool Hash_Table<T1,T2>::search(T1 key, T2 value){   
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        i = getHash<T1>(key,table_size,function1);
        
        if (hash_table1[i].value == value){
            return true;
        }else{
            return false;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+j)%table_size;
            if (hash_table1[i].value == value) return true;
            ++j;
        }
        return false;

    //Quadratische Hashverfahren
    }else if(type_hash == quadratic_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2) < table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            if (hash_table1[i].value == value) return true;
            ++j;
        }
        return false;

    //Doppelte Hashverfahren
    }else if (type_hash == double_probe){
        size_t i, j;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            if (hash_table1[i].value == value) return true;
            ++j;
        }
        return false;

    //Cuckoo-Hashverfahren    
    }else{
        size_t i = getHash<T1>(key,table_size,function1);
        size_t j = getHash<T1>(key,table_size,function2);
        size_t k = 1;

        if (hash_table1[i].value == value) return true;
        if (hash_table2[j].value == value) return true;

        while (k < table_size){
            i = (i + k) % table_size;
            j = (j + k) % table_size;
            
            if (hash_table1[i].value == value) return true;
            if (hash_table2[j].value == value) return true;

            ++k;
        }
        return false;
    }
};

/* Suche nach einem Array von Schlüsseln und deren Werten in einer von zwei Hashtabellen 
    gleichzeitig durch ausgewählte offene Hashverfahren:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
void Hash_Table<T1,T2>::search_List(T1 * keyList, T2 * valueList, size_t cellSize){
    if((cellSize > table_size) && type_hash != cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle einer Hashtabelle gesuchten Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe einer Hashtabelle betragen." << std::endl;

    }else if((cellSize > (2*table_size)) && type_hash == cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle von zwei Hashtabellen gesuchten Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe von zwei Hashtabellen betragen." << std::endl;

    }else{
        cell<T1,T2> * cells;
        T2 * keyListResult = new T2[cellSize];
        T2 * keyListResult_device;
        cell<T1,T2> * keyList_device;
        cell<T1,T2> * hash_table_device1;
        cell<T1,T2> * hash_table_device2;

        std::vector<cell<T1,T2>> cells_vector;
        
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
            cells_vector.push_back(cell<T1,T2>{keyList[i],valueList[i]});

        cells = cells_vector.data();

        if (dimension_kernel.num_blocks < 1 || dimension_kernel.num_threads_per_block < 1 ||
            cellSize != (size_t)(dimension_kernel.num_blocks*dimension_kernel.num_threads_per_block)){
            dimension_kernel.num_blocks = (int) cellSize;
            dimension_kernel.num_threads_per_block = 1;
        }

        dim3 num_Blocks(dimension_kernel.num_blocks,1,1);
        dim3 num_ThreadsPerBlock(dimension_kernel.num_threads_per_block,1,1);
        size_t shared_memory_size{dimension_kernel.num_threads_per_block*sizeof(cell<T1,T2>)};

        //Ohne Kollisionsauflösung, mit linearem und quadratischem Sondieren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T2)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T2)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();

            if (type_hash == linear_probe){
                search_linear<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                search_quadratic<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }else{
                search_normal<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1);
            }

            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T2)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Doppelte Hashverfahren
        }else if (type_hash == double_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T2)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T2)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            search_double<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, table_size, function1, function2);
            run.GPUstop(); 

            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T2)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();

            total.GPUstop();

        //Cuckoo-Hashverfahren
        }else{
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&hash_table_device2,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);
            cudaMalloc(&keyListResult_device,sizeof(T2)*cellSize);
            
            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyListResult_device,keyListResult,sizeof(T2)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            search_cuckoo<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, keyListResult_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
            run.GPUstop(); 
            
            //Kopiere Daten aus der GPU zur Hashtabelle
            download.GPUstart();
            cudaMemcpyAsync(keyListResult, keyListResult_device, sizeof(T2)*cellSize, cudaMemcpyDeviceToHost,download.getStream());
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

/* Löschung von Zellen in einer oder zwei Hashtabellen aus einem Paar 
    von einem Schlüssel und dessen Wert durch ausgewählte offene Hashverfahren:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
void Hash_Table<T1,T2>::deleteKey(T1 key, T2 value){
    //Ohne Kollisionsauflösung
    if (type_hash == no_probe){
        size_t i;
        T2 prev;

        i = getHash<T1>(key,table_size,function1);
        prev = swapHash<T2>(hash_table1[i].value, value, BLANK);

        if (prev == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].value = BLANK;
        }

    //Lineare Hashverfahren
    }else if(type_hash == linear_probe){
        size_t i, j;
        T2 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+j)%table_size;
            prev = swapHash<T2>(hash_table1[i].value, value, BLANK);

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
        T2 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while((j/2) < table_size){
            i = ((size_t) ((int) i + getProbe2(j))) %table_size;
            prev = swapHash<T2>(hash_table1[i].value, value, BLANK);

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
        T2 prev;
        
        i = getHash<T1>(key,table_size,function1);
        j = 0;

        while(j < table_size){
            i = (i+getHashProbe<T1>(key,j,table_size,function2))%table_size;
            prev = swapHash<T2>(hash_table1[i].value, value, BLANK);

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
        T2 prev1, prev2;

        i = getHash<T1>(key,table_size,function1);
        j = getHash<T1>(key,table_size,function2);
        k = 1;

        prev1 = swapHash<T2>(hash_table1[i].value, value, BLANK);

        if (prev1 == BLANK){
            hash_table1[i].key = BLANK;
            hash_table1[i].value = BLANK;
            return;
        }

        prev2 = swapHash<T2>(hash_table2[j].value, value, BLANK);

        if (prev2 == BLANK){
            hash_table2[j].key = BLANK;
            hash_table2[j].value = BLANK;
            return;
        }

        while (k < table_size){
            i = (i + k) % table_size;
            j = (j + k) % table_size;

            prev1 = swapHash<T2>(hash_table1[i].value, value, BLANK);

            if (prev1 == BLANK){
                hash_table1[i].key = BLANK;
                hash_table1[i].value = BLANK;
                break;
            }

            prev2 = swapHash<T2>(hash_table2[j].value, value, BLANK);

            if (prev2 == BLANK){
                hash_table2[j].key = BLANK;
                hash_table2[j].value = BLANK;
                break;
            }
            ++k;
        }
    }
};

/* Löschung von Zellen in einer oder zwei Hashtabellen aus einem Array von Schlüsseln und deren Werten 
    gleichzeitig durch ausgewählte offene Hashverfahren:
    a. ohne Kollisionsauflösung,
    b. linearem und quadratischem Sondieren,
    c. doppelten Hashverfahren oder
    d. Cuckoo-Hashverfahren
*/
template <typename T1, typename T2>
void Hash_Table<T1,T2>::delete_List(T1 * keyList, T2 * valueList, size_t cellSize){
    if((cellSize > table_size) && type_hash != cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle einer Hashtabelle zu löschenden Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe einer Hashtabelle betragen." << std::endl;

    }else if((cellSize > (2*table_size)) && type_hash == cuckoo_probe){
        std::cout << "Die Größe einer in einer Zelle von zwei Hashtabellen zu löschenden Zelle ";
        std::cout << "muss mindestens 0 und höchstens die Größe von zwei Hashtabellen betragen." << std::endl;

    }else{
        cell<T1,T2> * cells;
        cell<T1,T2> * keyList_device;
        cell<T1,T2> * hash_table_device1;
        cell<T1,T2> * hash_table_device2;

        std::vector<cell<T1,T2>> cells_vector;
        
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
            cells_vector.push_back(cell<T1,T2>{keyList[i],valueList[i]});

        cells = cells_vector.data();
                
        if (dimension_kernel.num_blocks < 1 || dimension_kernel.num_threads_per_block < 1 ||
            cellSize != (size_t)(dimension_kernel.num_blocks*dimension_kernel.num_threads_per_block)){
            dimension_kernel.num_blocks = (int) cellSize;
            dimension_kernel.num_threads_per_block = 1;
        }
        
        dim3 num_Blocks(dimension_kernel.num_blocks,1,1);
        dim3 num_ThreadsPerBlock(dimension_kernel.num_threads_per_block,1,1);
        size_t shared_memory_size{dimension_kernel.num_threads_per_block*sizeof(cell<T1,T2>)};

        //Ohne Kollisionsauflösung, mit linearem und quadratischem Sondieren
        if (type_hash == no_probe || type_hash == linear_probe || type_hash == quadratic_probe){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hash_table_device1,sizeof(cell<T1,T2>)*table_size);
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();

            if (type_hash == linear_probe){
                delete_linear<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }else if (type_hash == quadratic_probe){
                delete_quadratic<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }else{
                delete_normal<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1);
            }

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
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();

            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            delete_double<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, hash_table_device1, table_size, function1, function2);
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
            cudaMalloc(&keyList_device,sizeof(cell<T1,T2>)*cellSize);

            total.GPUstart();
            
            upload.GPUstart();
            cudaMemcpyAsync(hash_table_device1,hash_table1,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(hash_table_device2,hash_table2,sizeof(cell<T1,T2>)*table_size,cudaMemcpyHostToDevice,upload.getStream());
            cudaMemcpyAsync(keyList_device,cells,sizeof(cell<T1,T2>)*cellSize,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();
            
            //Lösche eine Liste von Schlüsseln in der Hashtabelle
            run.GPUstart();
            delete_cuckoo<T1,T2><<<num_Blocks,num_ThreadsPerBlock,shared_memory_size,run.getStream()>>>(keyList_device, hash_table_device1, hash_table_device2, table_size, function1, function2);
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

        num_cells_deleted = num_cells_prev - getNumCell();

        Benchmark_Delete.record(delete_hash_table,duration_upload,duration_run,duration_download,duration_total,num_cells_deleted,type_hash);
        benchmark_hash_table[2] = Benchmark_Delete;
    
        cudaFree(hash_table_device1);
        cudaFree(hash_table_device2);
        cudaFree(keyList_device);
    }
};

#endif