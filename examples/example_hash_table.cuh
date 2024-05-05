#ifndef EXAMPLE_HASH_TABLE_CUH
#define EXAMPLE_HASH_TABLE_CUH

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../core/hash_table.cuh>
#include <../tools/timer.cuh>
#include <../tools/benchmark.h>

#define Test_Num 100

template <typename T1, typename T2>
class Example_Hash_Table{
    private:
        std::vector<cell<T1,T2>> exampleCellOccupyList;
        std::vector<cell<T1,T2>> exampleCellList;

        size_t exampleCellOccupySize = 11;
        size_t exampleCellSize = 11;
        size_t exampleHashTableSize = exampleCellSize + 5;

        hash_function examplefunction1 = modulo;
        hash_function examplefunction2;

        kernel_dimension exampleKernelDimension;

        //Zeitmessung für Kernels
        struct Sum_Benchmark{
            operation_type OperationType;

            float upload{0};
            float run{0};
            float download{0};
            float total{0};

            size_t num_cells{0};

        } benchmark_kernel;

    public:
        //Konstruktor
        Example_Hash_Table(){
            exampleCellList.reserve(exampleCellSize);
        };
        
        Example_Hash_Table(size_t test_cell_size, size_t test_cell_occupy_size, size_t test_table_size,hash_function function1, hash_function function2=modulo,
                           int test_block_num = 0, int test_threads_num_per_block = 0):
        exampleCellSize(test_cell_size), exampleCellOccupySize(test_cell_occupy_size), exampleHashTableSize(test_table_size), examplefunction1(function1), examplefunction2(function2){
            if (test_cell_size>test_table_size || test_cell_occupy_size > test_table_size){
                std::cout << "Die Hashtabelle darf höchstens maximal " << test_table_size;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << test_cell_size << "und ";
                std::cout << "die Zahl der belegten Zellen " << test_cell_occupy_size << "." << std::endl;
                exit (EXIT_FAILURE);
            }

            if (test_block_num> 0 && test_threads_num_per_block > 0){
                exampleKernelDimension.num_blocks = test_block_num;
                exampleKernelDimension.num_threads_per_block = test_threads_num_per_block;
            }

            exampleCellSize = test_cell_size;
            exampleCellList.reserve(exampleCellSize);

            exampleCellOccupySize = test_cell_occupy_size;
            exampleCellOccupyList.reserve(exampleCellOccupySize);
        };

        //Destruktor
        ~Example_Hash_Table(){};

        //Gebe Schlüssel und deren Längen zurück
        std::vector<cell<T1,T2>> getKeys(){
            return exampleCellList;
        };

        //Lese eine Liste von Schlüsseln von einer Datei
        void readCells(char *file_name){
            std::ifstream readfile(file_name, std::ios::out | std::ios::binary);
            
            if (!readfile) {
                std::cout << "Die Datei wird nicht gefunden." << std::endl;
                return exit(EXIT_FAILURE);
            }

            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
    
            std::vector<cell<T1,T2>> cells_vector;
            cell<T1,T2> key;

            cells_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize; i++){
                readfile.read((char*) &key, sizeof(cell<T1,T2>));
                cells_vector.push_back(cell<T1,T2>{key.key,key.value});
            }
            
            readfile.close();
            
            std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
            std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());       
        };

        //Erzeuge verschiedene Werte für die Schlüssel und deren Werten zufällig für Auslastungsfaktor
        void createOccupyCells(){
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::vector<cell<T1,T2>> cells_vector;
            cells_vector.reserve(exampleCellOccupySize);

            std::uniform_int_distribution<T1> dist(1,exampleCellOccupySize);

            T1 key = 1;
            T2 value = dist(rnd); 
        
            for (size_t i = 0; i < exampleCellOccupySize; i++){    
                cells_vector.push_back(cell<T1,T2>{key,value});
                ++key;
            }          
            std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
            std::copy(cells_vector.begin(),cells_vector.end(),exampleCellOccupyList.begin());
        };

        //Erzeuge verschiedene Werte für die Schlüssel und deren Werten zufällig
        void createCells(bool value_same = false){
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::vector<cell<T1,T2>> cells_vector;
            cells_vector.reserve(exampleCellSize);

            if (value_same == false){
                T1 key = 0; 
                T2 value = 0;
        
                for (size_t i = 0; i < exampleCellSize; i++){
                    ++key;
                    ++value;
                    cells_vector.push_back(cell<T1,T2>{key,value});
                }                
                std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
                std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());

            }else{
                std::uniform_int_distribution<T1> dist(1,exampleCellSize);

                T1 key = 0;
                T2 value = dist(rnd); 
        
                for (size_t i = 0; i < exampleCellSize; i++){
                    ++key;
                    cells_vector.push_back(cell<T1,T2>{key,value});
                }          
                std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
                std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());
            }
        };

        //Mische verschiedene Schlüssel
        void shuffleKeys(){
            std::vector<cell<T1,T2>> cells_vector;
            cells_vector.reserve(exampleCellList.size());
    
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
    
            std::copy(exampleCellList.begin(), exampleCellList.end(),cells_vector.begin());
            std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
    
            std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Sequentielle und parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten 
        //bei einem Versuch hinzu    
        void insertTestCells1(hash_type HashType){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            cell<T1,T2> * cellArray;
            cell<T1,T2> * cellOccupyArray;
            T1 * keyListArray;
            T1 * keyListOccupyArray;
            T2 * valueListArray;
            T2 * valueListOccupyArray;
            std::vector<T1> key_vector, key_occupy_vector;
            std::vector<T2> value_vector, value_occupy_vector;
            
            createOccupyCells();

            cellArray = exampleCellList.data();
            cellOccupyArray = exampleCellOccupyList.data();
            
            key_vector.reserve(exampleCellSize);
            key_occupy_vector.reserve(exampleCellOccupySize);      
            value_vector.reserve(exampleCellSize);
            value_occupy_vector.reserve(exampleCellOccupySize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }

            for (size_t i = 0; i < exampleCellOccupySize ; i++){
                key_occupy_vector.push_back(cellOccupyArray[i].key);
                value_occupy_vector.push_back(cellOccupyArray[i].value);
            }
            
            keyListArray = key_vector.data();
            keyListOccupyArray = key_occupy_vector.data();
            valueListArray = value_vector.data();
            valueListOccupyArray = value_occupy_vector.data();
            
            if (HashType == no_probe){
                std::cout << "OHNE KOLLISIONSAUFLÖSUNG" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "LINEARES SONDIEREN" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "QUADRATISCHES SONDIEREN" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "DOPPELTE HASHVERFAHREN" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "CUCKOO-HASHVERFAHREN" << std::endl;
            }
            std::cout << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            Hash_Table<T1,T2> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            CPUTimer timer;
            timer.start();
            hash_table1.insert_List(key_occupy_vector.data(), value_occupy_vector.data(), exampleCellOccupySize);
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            timer.stop();

            hash_table1.print();

            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << std::endl;
            std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize, 
                                      exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.insert_List(key_occupy_vector.data(), value_occupy_vector.data(), exampleCellOccupySize);

            hash_table2.insert_List(key_vector.data(), value_vector.data(), exampleCellSize);
            Benchmark Benchmark_Insert = hash_table2.getBenchmark(insert_hash_table);
            
            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << "," << "Run_Dauer" << ",";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "ZahlGespeichert"<< std::endl;
            Benchmark_Insert.print();

            std::cout << std::endl;    
            hash_table2.print();
            std::cout << std::endl;    

            size_t numCells1 = 0;
            size_t numCells2 = 0;

            numCells1 = hash_table1.getNumCell();
            numCells2 = hash_table2.getNumCell();
            
            if (numCells1 == numCells2){
                std::cout << "Anzahl der Zellen in der Hashtabelle bei" << "," << numCells1 << std::endl;
                std::cout << "sequentiellen und parallelen Ausführungen" << std::endl;
                std::cout << std::endl;
            }else{
                std::cout << "Anzahl der Zellen in der Hashtabelle bei" << std::endl;
                std::cout << "a) sequentiellen Ausführungen" << "," << numCells1 << std::endl;
                std::cout << "b) parallelen Ausführungen" << "," << numCells2 << std::endl;
                std::cout << std::endl;
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten 
        //bei Test_Num Versuchen, d.h 100 Versuchen hinzu       
        void insertTestCells2(hash_type HashType){
            cell<T1,T2> * cellArray;
            cell<T1,T2> * cellOccupyArray;
            T1 * keyListArray;
            T1 * keyListOccupyArray;
            T2 * valueListArray;
            T2 * valueListOccupyArray;
            std::vector<T1> key_vector, key_occupy_vector;
            std::vector<T2> value_vector, value_occupy_vector;
            
            std::string HashTypeString;
            float averageDurationUpload, averageDurationRun, averageDurationDownload, averageDurationTotal, averageNumCellsInsert;

            createOccupyCells();

            cellArray = exampleCellList.data();
            cellOccupyArray = exampleCellOccupyList.data();
            
            key_vector.reserve(exampleCellSize);
            key_occupy_vector.reserve(exampleCellOccupySize);      
            value_vector.reserve(exampleCellSize);
            value_occupy_vector.reserve(exampleCellOccupySize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }

            for (size_t i = 0; i < exampleCellOccupySize ; i++){
                key_occupy_vector.push_back(cellOccupyArray[i].key);
                value_occupy_vector.push_back(cellOccupyArray[i].value);
            }
            
            keyListArray = key_vector.data();
            keyListOccupyArray = key_occupy_vector.data();
            valueListArray = value_vector.data();
            valueListOccupyArray = value_occupy_vector.data();

            benchmark_kernel.OperationType = insert_hash_table;

            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << ", " << "Run_Dauer" << ",";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "ZahlGespeichert"<< std::endl;

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize, 
                                          exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);
                hash_table2.insert_List(key_occupy_vector.data(), value_occupy_vector.data(), exampleCellOccupySize);
                
                hash_table2.insert_List(key_vector.data(), value_vector.data(), exampleCellSize);
                Benchmark Benchmark_Insert = hash_table2.getBenchmark(insert_hash_table);
                
                Benchmark_Insert.print();

                benchmark_kernel.upload+=Benchmark_Insert.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Insert.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Insert.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Insert.getDurationTotal();

                benchmark_kernel.num_cells+=Benchmark_Insert.getNumCells();
            }
            std::cout << std::endl;

            averageDurationUpload = benchmark_kernel.upload/Test_Num;
            averageDurationRun = benchmark_kernel.run/Test_Num;
            averageDurationDownload = benchmark_kernel.download/Test_Num;
            averageDurationTotal = benchmark_kernel.total/Test_Num;
            averageNumCellsInsert = ((float) benchmark_kernel.num_cells)/Test_Num;

            std::cout << "Kernel_Name" << "," << "Zahl_Versuche" << std::endl;

            if (HashType == linear_probe){
                HashTypeString.append("insert_linear<>");    
            }else if (HashType == quadratic_probe){
                HashTypeString.append("insert_quadratic<>");
            }else if (HashType == double_probe){
                HashTypeString.append("insert_double<>");   
            }else if (HashType == cuckoo_probe){
                HashTypeString.append("insert_cuckoo<>");
            }else{
                HashTypeString.append("insert_normal<>");    
            }
            std::cout << HashTypeString << ", " << Test_Num << std::endl;

            std::cout << "Upload_Gesamtdauer" << "," << "Run_Gesamtdauer" << "," << "Download_Gesamtdauer" << ",";
            std::cout << "Total_Gesamtdauer" <<  "," << "Gesamt_Gespeichert" << std::endl;
            std::cout <<  benchmark_kernel.upload << "," << benchmark_kernel.run << "," <<  benchmark_kernel.download << ",";
            std::cout << benchmark_kernel.total << "," << benchmark_kernel.num_cells << std::endl;
            std::cout << "Upload_Durchschnittsdauer" << "," << "Run_Durchschnittsdauer" << "," << "Download_Durchschnittsdauer" << ",";
            std::cout << "Total_Durchschnittsdauer" << "," << "Durchschnitt_Gespeichert" << std::endl;
            std::cout <<  averageDurationUpload << "," << averageDurationRun << "," <<  averageDurationDownload << ",";
            std::cout << averageDurationTotal << "," << averageNumCellsInsert << std::endl;
            std::cout << std::endl;
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Sequentielle und parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Suche nach einer Liste von Schlüsseln in der Hashtabelle bei einem Versuch   
        void searchTestCells1(hash_type HashType){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t sum_found = 0;

            cell<T1,T2> * cellArray;
            T1 * keyListArray;
            T2 * valueListArray;
            std::vector<T1> key_vector;
            std::vector<T2> value_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize);
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key); 
                value_vector.push_back(cellArray[i].value);
            }

            keyListArray = key_vector.data();
            valueListArray = value_vector.data();

            Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize,
                                     exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);

            if (HashType == no_probe){
                std::cout << "OHNE KOLLISIONSAUFLÖSUNG" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "LINEARES SONDIEREN" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "QUADRATISCHES SONDIEREN" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "DOPPELTE HASHVERFAHREN" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "CUCKOO-HASHVERFAHREN" << std::endl;
            }
            std::cout << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
            
            shuffleKeys();

            cellArray = exampleCellList.data();

            key_vector.clear();
            value_vector.clear();

            key_vector.reserve(exampleCellSize);
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }

            keyListArray = key_vector.data();
            valueListArray = value_vector.data();           

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) if (hash_table.search(keyListArray[i],valueListArray[i]) == true) ++sum_found;
            timer.stop();

            hash_table.print();

            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Anzahl der gesuchten Zellen" << "," << sum_found << std::endl;
            std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table.search_List(key_vector.data(), value_vector.data(), exampleCellSize);
            Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);

            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << "," << "Run_Dauer" << ",";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "SummeGefunden"<< std::endl;
            Benchmark_Search.print();

            std::cout << std::endl;
            hash_table.print();
            std::cout << std::endl;
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Suche nach einer Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen   
        void searchTestCells2(hash_type HashType){
            cell<T1,T2> * cellArray;
            T1 * keyListArray;
            T2 * valueListArray;
            std::vector<T1> key_vector;
            std::vector<T2> value_vector;

            std::string HashTypeString;
            float averageDurationUpload, averageDurationRun, averageDurationDownload, averageDurationTotal,averageSumFound;
            
            cellArray = exampleCellList.data();

            key_vector.reserve(exampleCellSize);           
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }

            keyListArray = key_vector.data();
            valueListArray = value_vector.data();            

            benchmark_kernel.OperationType = search_hash_table;

            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << "," << "Run_Dauer" << ",";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "SummeGefunden"<< std::endl;

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize,
                                         exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
                shuffleKeys();

                hash_table.search_List(key_vector.data(), value_vector.data(), exampleCellSize);
                Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);
                
                Benchmark_Search.print();
         
                benchmark_kernel.upload+=Benchmark_Search.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Search.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Search.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Search.getDurationTotal();
                
                benchmark_kernel.num_cells+=Benchmark_Search.getNumCells();
            }

            std::cout << std::endl;

            averageDurationUpload = benchmark_kernel.upload/Test_Num;
            averageDurationRun = benchmark_kernel.run/Test_Num;
            averageDurationDownload = benchmark_kernel.download/Test_Num;
            averageDurationTotal = benchmark_kernel.total/Test_Num;
            averageSumFound = ((float)benchmark_kernel.num_cells)/Test_Num;

            std::cout << "Kernel_Name" << "," << "Zahl_Versuche" << std::endl;

            if (HashType == linear_probe){
                HashTypeString.append("search_linear<>");    
            }else if (HashType == quadratic_probe){
                HashTypeString.append("search_quadratic<>");
            }else if (HashType == double_probe){
                HashTypeString.append("search_double<>");   
            }else if (HashType == cuckoo_probe){
                HashTypeString.append("search_cuckoo<>");
            }else{
                HashTypeString.append("search_normal<>");    
            }

            std::cout << HashTypeString << "," << Test_Num << "," << std::endl;
            
            std::cout << "Upload_Gesamtdauer" << "," << "Run_Gesamtdauer" << "," << "Download_Gesamtdauer" << ",";
            std::cout << "Total_Gesamtdauer" << "," << "Gesamt_Gefunden" << std::endl;
            std::cout <<  benchmark_kernel.upload << "," << benchmark_kernel.run << "," <<  benchmark_kernel.download << ",";
            std::cout << benchmark_kernel.total << "," << benchmark_kernel.num_cells << std::endl;
            std::cout << "Upload_Durchschnittsdauer" << "," << "Run_Durchschnittsdauer" << "," << "Download_Durchschnittsdauer" << ","; 
            std::cout << "Total_Durchschnittsdauer" << "," << "Durchschnitt_Gefunden" << std::endl;
            std::cout <<  averageDurationUpload << "," << averageDurationRun << "," <<  averageDurationDownload << ",";
            std::cout << averageDurationTotal << "," << averageSumFound << std::endl;
            std::cout << std::endl;
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Sequentielle und parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle bei einem Versuch
        void deleteTestCells1(hash_type HashType, bool value_same = false){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t num_cells_prev1, num_cells_deleted1;
            size_t num_cells_deleted2;

            cell<T1,T2> * cellArray;
            T1 * keyListArray;
            T2 * valueListArray;
            std::vector<T1> key_vector;
            std::vector<T2> value_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize);           
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }
            
            keyListArray = key_vector.data();
            valueListArray = value_vector.data();

            Hash_Table<T1,T2> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            if (HashType == no_probe){
                std::cout << "OHNE KOLLISIONSAUFLÖSUNG" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "LINEARES SONDIEREN" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "QUADRATISCHES SONDIEREN" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "DOPPELTE HASHVERFAHREN" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "CUCKOO-HASHVERFAHREN" << std::endl;
            }
            std::cout << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            num_cells_prev1 = hash_table1.getNumCell();
            createCells(value_same);
            
            cellArray = exampleCellList.data();

            key_vector.clear();
            value_vector.clear();

            key_vector.reserve(exampleCellSize);
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }
            
            keyListArray = key_vector.data();
            valueListArray = value_vector.data();

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.deleteKey(keyListArray[i],valueListArray[i]);
            timer.stop();

            hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize,
                                      exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);
            for (size_t i=0; i<exampleCellSize; i++) hash_table2.insert(keyListArray[i],valueListArray[i]);

            shuffleKeys();
            
            cellArray = exampleCellList.data();

            key_vector.clear();
            value_vector.clear();

            key_vector.reserve(exampleCellSize);
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }
            
            keyListArray = key_vector.data();
            valueListArray = value_vector.data();

            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.delete_List(key_vector.data(), value_vector.data(), exampleCellSize);
            Benchmark Benchmark_Delete = hash_table2.getBenchmark(delete_hash_table);

            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << "," << "Run_Dauer" << ", ";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "Zellen_Gelöscht"<< std::endl;
            Benchmark_Delete.print();

            std::cout << std::endl;
            hash_table2.print();
            std::cout << std::endl;

            num_cells_deleted1 = num_cells_prev1 - hash_table1.getNumCell();
            num_cells_deleted2 = Benchmark_Delete.getNumCells();

            if (num_cells_deleted1 == num_cells_deleted2){
                std::cout << "Anzahl der gelöschten Zellen bei sequentiellen und parallelen Ausführungen" << ",";
                std::cout << num_cells_deleted1 << std::endl;
                std::cout << std::endl;
            }else{
                std::cout << "Anzahl der gelöschten Zellen in der Hashtabelle bei" << std::endl;
                std::cout << "a) sequentiellen Ausführungen" << "," << num_cells_deleted1 << std::endl;
                std::cout << "b) parallelen Ausführungen" << "," << num_cells_deleted2 << std::endl;
                std::cout << std::endl;
            }
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen    
        void deleteTestCells2(hash_type HashType, bool value_same = false){
            cell<T1,T2> * cellArray;
            T1 * keyListArray;
            T2 * valueListArray;
            std::vector<T1> key_vector;
            std::vector<T2> value_vector;
            
            std::string HashTypeString;
            float averageDurationUpload, averageDurationRun, averageDurationDownload, averageDurationTotal, averageDeletedCells;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize);            
            value_vector.reserve(exampleCellSize); 

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                value_vector.push_back(cellArray[i].value);
            }
            
            keyListArray = key_vector.data();
            valueListArray = value_vector.data();

            benchmark_kernel.OperationType = delete_hash_table;

            std::cout << "Kernel_Name" << "," << "Upload_Dauer" << "," << "Run_Dauer" << ",";
            std::cout << "Download_Dauer" << "," << "Total_Dauer" << "," << "Zellen_Gelöscht"<< std::endl;

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize,
                                         exampleKernelDimension.num_blocks, exampleKernelDimension.num_threads_per_block);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);

                createCells(value_same);
                
                hash_table.delete_List(key_vector.data(), value_vector.data(), exampleCellSize);
                Benchmark Benchmark_Delete = hash_table.getBenchmark(delete_hash_table);

                Benchmark_Delete.print();
 
                benchmark_kernel.upload+=Benchmark_Delete.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Delete.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Delete.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Delete.getDurationTotal();
               
                benchmark_kernel.num_cells+=Benchmark_Delete.getNumCells();
            }
            std::cout << std::endl;
            
            averageDurationUpload = benchmark_kernel.upload/Test_Num;
            averageDurationRun = benchmark_kernel.run/Test_Num;
            averageDurationDownload = benchmark_kernel.download/Test_Num;
            averageDurationTotal = benchmark_kernel.total/Test_Num;
            averageDeletedCells = ((float)benchmark_kernel.num_cells)/Test_Num;

            std::cout << "Kernel_Name" << "," << "Zahl_Versuche" << std::endl;

            if (HashType == linear_probe){
                HashTypeString.append("delete_linear<>");    
            }else if (HashType == quadratic_probe){
                HashTypeString.append("delete_quadratic<>");
            }else if (HashType == double_probe){
                HashTypeString.append("delete_double<>");   
            }else if (HashType == cuckoo_probe){
                HashTypeString.append("delete_cuckoo<>");
            }else{
                HashTypeString.append("delete_normal<>");    
            }

            std::cout << HashTypeString << "," << Test_Num << std::endl;
            
            std::cout << "Upload_Gesamtdauer" << "," << "Run_Gesamtdauer" << ","<< "Download_Gesamtdauer" << ",";
            std::cout << "Total_Gesamtdauer" << "," << "Gesamt_Gelöscht" << std::endl;
            std::cout <<  benchmark_kernel.upload << "," << benchmark_kernel.run << "," <<  benchmark_kernel.download << ",";
            std::cout << benchmark_kernel.total << "," << benchmark_kernel.num_cells << std::endl;
            std::cout << "Upload_Durchschnittsdauer" << "," << "Run_Durchschnittsdauer" << "," << "Download_Durchschnittsdauer" << ",";
            std::cout << "Total_Durchschnittsdauer" << "," << "Durchschnitt_Gelöscht" << std::endl;
            std::cout <<  averageDurationUpload << "," << averageDurationRun << "," <<  averageDurationDownload << ",";
            std::cout << averageDurationTotal << "," << averageDeletedCells << std::endl;
            std::cout << std::endl;
        };
};


//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

// //1. Deklariere die Variablen
// int i_inBlock, blockID, i;
// size_t j;
// T key, prev, value; 

// //2. Bestimme die globale ID eines Threads
// i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
// blockID = blockIdx.x;
// i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

#endif