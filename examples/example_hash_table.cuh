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

template <typename T>
class Example_Hash_Table{
    private:
        std::vector<cell<T>> exampleCellList;

        size_t exampleCellSize = 11;
        size_t exampleHashTableSize = exampleCellSize + 5;

        hash_function examplefunction1 = modulo;
        hash_function examplefunction2;

        //Zeitmessung für Kernels
        struct Sum_Benchmark{
            operation_type OperationType;

            float upload{0};
            float run{0};
            float download{0};
            float total{0};

            size_t SumFound{0};
            size_t sum_num_deleted_cells{0};

        } benchmark_kernel;

    public:
        //Konstruktor
        Example_Hash_Table(){
            exampleCellList.reserve(exampleCellSize);
        };
        
        Example_Hash_Table(size_t test_cell_size, size_t test_table_size,hash_function function1, hash_function function2=modulo):
        exampleCellSize(test_cell_size),exampleHashTableSize(test_table_size),examplefunction1(function1),examplefunction2(function2){
            if (test_cell_size>test_table_size){
                std::cout << "Die Hashtabelle darf höchstens maximal " << test_table_size;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << test_cell_size<< "." << std::endl;
                exit (EXIT_FAILURE);
            }
            exampleCellList.reserve(test_cell_size);
        };

        //Destruktor
        ~Example_Hash_Table(){};

        //Gebe Schlüssel und deren Längen zurück
        std::vector<cell<T>> getKeys(){
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
    
            std::vector<cell<T>> cells_vector;
            cell<T> key;

            cells_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize; i++){
                readfile.read((char*) &key, sizeof(cell<T>));
                cells_vector.push_back(cell<T>{key.key,key.key_length});
            }
            
            readfile.close();
            
            std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
            std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());       
        };

        //Erzeuge verschiedene Werte für die Schlüssel und deren Längen zufällig
        void createCells(bool key_length_same = false){
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::vector<cell<T>> cells_vector;
            cells_vector.reserve(exampleCellSize);

            if (key_length_same == false){
                T key = 0;
                T key_length = 0; 
        
                for (size_t i = 0; i < exampleCellSize; i++){
                    ++key;
                    ++key_length;
                    cells_vector.push_back(cell<T>{key,key_length});
                }                
                std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
                std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());

            }else{
                std::uniform_int_distribution<T> dist(1,exampleCellSize);

                T key = 0;
                T key_length = dist(rnd); 
        
                for (size_t i = 0; i < exampleCellSize; i++){
                    ++key;
                    cells_vector.push_back(cell<T>{key,key_length});
                }          
                std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
                std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());
            }
        };

        //Mische verschiedene Schlüssel
        void shuffleKeys(){
            std::vector<cell<T>> cells_vector;
            cells_vector.reserve(exampleCellList.size());
    
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
    
            std::copy(exampleCellList.begin(), exampleCellList.end(),cells_vector.begin());
            std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
    
            std::copy(cells_vector.begin(),cells_vector.end(),exampleCellList.begin());
        };

        void printBenchmark(float Upload, float Run, float Download, float Total,
                            float SumFound=0, float DeletedCells=0){
            if (benchmark_kernel.OperationType == insert_hash_table){
                std::cout << "insert_hash_table" << ", " << Upload;
                std::cout << ", " << Run << ", " << Download  << ", ";
                std::cout <<  Total  << std::endl;

            }else if (benchmark_kernel.OperationType == search_hash_table){
                std::cout << "search_hash_table" << ", " << Upload ;
                std::cout << ", " << Run << ", " << Download  << ", ";
                std::cout <<  Total << "," << SumFound << std::endl;
        
            }else{
                std::cout << "delete_hash_table" << ", " << Upload;
                std::cout << ", " << Run << ", " << Download << ", ";
                std::cout <<  Total << "," << DeletedCells << std::endl;
            }
        };

        //Drucke Gesamtzeitmessung für den Kernel
        void printTotalBenchmark(){
            if (benchmark_kernel.OperationType == insert_hash_table){
                std::cout << "Speicherung von Schlüsseln in der Hashtabelle bei ";
                std::cout << Test_Num << " Versuchen" << std::endl;
                std::cout << std::endl;

            }else if (benchmark_kernel.OperationType == search_hash_table){
                size_t averageSumFound = benchmark_kernel.SumFound/Test_Num;
                std::cout << "Suche nach Schlüsseln in der Hashtabelle bei ";
                std::cout << Test_Num << " Versuchen" << std::endl;
                std::cout << std::endl;
                std::cout << "Gesamtanzahl der gesuchten Zellen           : ";
                std::cout << benchmark_kernel.SumFound << std::endl;
                std::cout << "Durchschnitt der gesuchten Zellen           : ";
                std::cout << averageSumFound << std::endl;
                std::cout << std::endl;

            }else{
                size_t averageDeletedCells = benchmark_kernel.sum_num_deleted_cells/Test_Num;
                std::cout << "Löschung von Schlüsseln in der Hashtabelle bei ";
                std::cout << Test_Num << " Versuchen" << std::endl;
                std::cout << std::endl;
                std::cout << "Gesamtanzahl der gelöschten Zellen          : ";
                std::cout << benchmark_kernel.sum_num_deleted_cells << std::endl;
                std::cout << "Anzahl der gelöschten Zellen                : ";
                std::cout << averageDeletedCells << std::endl;
                std::cout << std::endl;
            }

            float averageDurationUpload, averageDurationRun, averageDurationDownload, averageDurationTotal;

            averageDurationUpload = benchmark_kernel.upload/Test_Num;
            averageDurationRun = benchmark_kernel.run/Test_Num;
            averageDurationDownload = benchmark_kernel.download/Test_Num;
            averageDurationTotal = benchmark_kernel.total/Test_Num;

            std::cout << "Gesamtdauer zum Hochladen                   : ";
            std::cout <<  benchmark_kernel.upload << std::endl;
            std::cout << "Gesamtdauer zur Ausführung                  : ";
            std::cout <<  benchmark_kernel.run << std::endl;
            std::cout << "Gesamtdauer zum Herunterladen               : ";
            std::cout <<  benchmark_kernel.download << std::endl;
            std::cout << "Gesamtdauer                                 : ";
            std::cout <<  benchmark_kernel.total << std::endl;
            std::cout << std::endl;

            std::cout << "Durchschnittsdauer zum Hochladen            : ";
            std::cout <<  averageDurationUpload << std::endl;
            std::cout << "Durchschnittsdauer zur Ausführung           : ";
            std::cout <<  averageDurationRun << std::endl;
            std::cout << "Durchschnittsdauer zum Herunterladen        : ";
            std::cout <<  averageDurationDownload << std::endl;
            std::cout << "Gesamtdurchschnittsdauer                    : ";
            std::cout <<  averageDurationTotal << std::endl;
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
            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();
        
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            Hash_Table<T> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],keyLengthListArray[i]);
            timer.stop();

            hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Gesamtdauer                                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.insert_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
            Benchmark Benchmark_Insert = hash_table2.getBenchmark(insert_hash_table);
            Benchmark_Insert.print();

            size_t numCells1 = 0;
            size_t numCells2 = 0;

            numCells1 = hash_table1.getNumCell();
            numCells2 = hash_table2.getNumCell();
            
            if (numCells1 == numCells2){
                std::cout << std::endl;
                std::cout << "Anzahl der Zellen in der Hashtabelle bei    : ";
                std::cout << numCells1 << std::endl;
                std::cout << "sequentiellen und parallelen Ausführungen" << std::endl;
                std::cout << std::endl;
            }else{
                std::cout << std::endl;
                std::cout << "Anzahl der Zellen in der Hashtabelle bei" << std::endl;
                std::cout << "a) sequentiellen Ausführungen               : ";
                std::cout << numCells1 << std::endl;
                std::cout << "b) parallelen Ausführungen                  : ";
                std::cout << numCells2 << std::endl;
                std::cout << std::endl;
            }
        
            hash_table2.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten 
        //bei Test_Num Versuchen, d.h 100 Versuchen hinzu       
        void insertTestCells2(hash_type HashType){
            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            benchmark_kernel.OperationType = insert_hash_table;

            std::cout << "OperationType" << ", " << "Upload_Duration" << ", " << "Run_Duration" << ", ";
            std::cout << "Download_Duration" << ", " << "Total_Duration" << std::endl;

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                hash_table2.insert_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
                Benchmark Benchmark_Insert = hash_table2.getBenchmark(insert_hash_table);
                
                printBenchmark(Benchmark_Insert.getDurationUpload(),Benchmark_Insert.getDurationRun(),
                               Benchmark_Insert.getDurationDownload(),Benchmark_Insert.getDurationTotal());
         
                benchmark_kernel.upload+=Benchmark_Insert.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Insert.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Insert.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Insert.getDurationTotal();
            }
            std::cout << std::endl;
            printTotalBenchmark();
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

            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            Hash_Table<T> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],keyLengthListArray[i]);
            
            shuffleKeys();

            cellArray = exampleCellList.data();

            key_vector.clear();
            key_length_vector.clear();

            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) if (hash_table.search(keyListArray[i],keyLengthListArray[i]) == true) ++sum_found;
            timer.stop();

            hash_table.print();

            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Anzahl der gesuchten Zellen                 : ";
            std::cout << sum_found << std::endl;
            std::cout << "Gesamtdauer                                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table.search_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
            Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);
            Benchmark_Search.print();
            hash_table.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Suche nach einer Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen   
        void searchTestCells2(hash_type HashType){
            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            benchmark_kernel.OperationType = search_hash_table;

            std::cout << "OperationType" << ", " << "Upload_Duration" << ", " << "Run_Duration" << ", ";
            std::cout << "Download_Duration" << ", " << "Total_Duration" << ", " << "SumFound"<< std::endl;

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],keyLengthListArray[i]);
                shuffleKeys();

                hash_table.search_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
                Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);
                
                printBenchmark(Benchmark_Search.getDurationUpload(),Benchmark_Search.getDurationRun(),Benchmark_Search.getDurationDownload(),
                               Benchmark_Search.getDurationTotal(),Benchmark_Search.getSumFound());
         
                benchmark_kernel.upload+=Benchmark_Search.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Search.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Search.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Search.getDurationTotal();
                
                benchmark_kernel.SumFound+=Benchmark_Search.getSumFound();
            }
            std::cout << std::endl;
            printTotalBenchmark();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Sequentielle und parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle bei einem Versuch
        void deleteTestCells1(hash_type HashType, bool key_length_same = false){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t num_cells_prev1, num_cells_curr1;
            size_t num_cells_prev2, num_cells_curr2;

            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            Hash_Table<T> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],keyLengthListArray[i]);
            num_cells_prev1 = hash_table1.getNumCell();
            createCells(key_length_same);
            
            cellArray = exampleCellList.data();

            key_vector.clear();
            key_length_vector.clear();

            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.deleteKey(keyListArray[i],keyLengthListArray[i]);
            timer.stop();

            hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Gesamtdauer                                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
            for (size_t i=0; i<exampleCellSize; i++) hash_table2.insert(keyListArray[i],keyLengthListArray[i]);
            num_cells_prev2 = hash_table2.getNumCell();

            shuffleKeys();
            
            cellArray = exampleCellList.data();

            key_vector.clear();
            key_length_vector.clear();

            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.delete_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
            Benchmark Benchmark_Delete = hash_table2.getBenchmark(delete_hash_table);
            Benchmark_Delete.print();

            num_cells_curr1 = num_cells_prev1 - hash_table1.getNumCell();
            num_cells_curr2 = num_cells_prev2 - hash_table2.getNumCell();

            if (num_cells_curr1 == num_cells_curr2){
                std::cout << std::endl;
                std::cout << "Anzahl der gelöschten Zellen bei            : ";
                std::cout << num_cells_curr1 << std::endl;
                std::cout << "sequentiellen und parallelen Ausführungen" << std::endl;
                std::cout << std::endl;
            }else{
                std::cout << std::endl;
                std::cout << "Anzahl der gelöschten Zellen in der Hashtabelle bei" << std::endl;
                std::cout << "a) sequentiellen Ausführungen               : ";
                std::cout << num_cells_curr1 << std::endl;
                std::cout << "b) parallelen Ausführungen                  : ";
                std::cout << num_cells_curr2 << std::endl;
                std::cout << std::endl;
            }

            hash_table2.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen    
        void deleteTestCells2(hash_type HashType, bool key_length_same = false){
            cell<T> * cellArray;
            T * keyListArray;
            T * keyLengthListArray;
            std::vector<T> key_vector, key_length_vector;
            
            cellArray = exampleCellList.data();
            
            key_vector.reserve(exampleCellSize); 
            key_length_vector.reserve(exampleCellSize);

            for (size_t i = 0; i < exampleCellSize ; i++){
                key_vector.push_back(cellArray[i].key);
                key_length_vector.push_back(cellArray[i].key_length);
            }
            
            keyListArray = key_vector.data();
            keyLengthListArray = key_length_vector.data();

            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (HashType == no_probe){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(HashType == linear_probe){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(HashType== quadratic_probe){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(HashType == double_probe){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }else if(HashType == cuckoo_probe){
                std::cout << "Cuckoo-Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            benchmark_kernel.OperationType = delete_hash_table;

            std::cout << "OperationType" << ", " << "Upload_Duration" << ", " << "Run_Duration" << ", ";
            std::cout << "Download_Duration" << ", " << "Total_Duration" << ", " << "DeletedCells"<< std::endl;

            for (int i = 0; i < Test_Num; i++){
                size_t num_cells_prev, num_cells_curr;

                Hash_Table<T> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],keyLengthListArray[i]);
                num_cells_prev = hash_table.getNumCell();

                createCells(key_length_same);
                
                hash_table.delete_List(key_vector.data(),key_length_vector.data(),exampleCellSize);
                Benchmark Benchmark_Delete = hash_table.getBenchmark(delete_hash_table);
                num_cells_curr = num_cells_prev - hash_table.getNumCell();

                printBenchmark(Benchmark_Delete.getDurationUpload(),Benchmark_Delete.getDurationRun(),Benchmark_Delete.getDurationDownload(),
                               Benchmark_Delete.getDurationTotal(),num_cells_curr);

                benchmark_kernel.upload+=Benchmark_Delete.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Delete.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Delete.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Delete.getDurationTotal();
                
                benchmark_kernel.sum_num_deleted_cells+=num_cells_curr;
            }
            std::cout << std::endl;
            printTotalBenchmark();
        };
};

#endif