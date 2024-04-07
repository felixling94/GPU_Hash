#ifndef EXAMPLE_HASH_TABLE_CUH
#define EXAMPLE_HASH_TABLE_CUH

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../core/hash_table.cuh>
#include <../tools/timer.cuh>
#include <../tools/benchmark.h>

#define Test_Num 100

template <typename T1, typename T2>
class Example_Hash_Table{
    private:
        std::vector<T1> exampleKeyList;
        std::vector<T2> exampleValueList;

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
            exampleKeyList.reserve(exampleCellSize);
            exampleValueList.reserve(exampleCellSize);
        };
        
        Example_Hash_Table(size_t test_cell_size, size_t test_table_size,hash_function function1, hash_function function2=modulo):
        exampleCellSize(test_cell_size),exampleHashTableSize(test_table_size),examplefunction1(function1),examplefunction2(function2){
            if (test_cell_size>test_table_size){
                std::cout << "Die Hashtabelle darf höchstens maximal " << test_table_size;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << test_cell_size<< "." << std::endl;
                exit (EXIT_FAILURE);
            }
            
            exampleKeyList.reserve(test_cell_size);
            exampleValueList.reserve(test_cell_size);
        };

        //Destruktor
        ~Example_Hash_Table(){};

        //Gebe Schlüssel zurück
        std::vector<T1> getKeys(){
            return exampleKeyList;
        };

        //Gebe Werte zurück
        std::vector<T2> getKValues(){
            return exampleValueList;
        };

        //Erzeuge verschiedene Werte für die Schlüssel und Werte zufällig
        void createCells(int min=0, int max=100){
            std::vector<T1> keys_vector;
            std::vector<T2> values_vector;
            
            keys_vector.reserve(exampleCellSize);
            values_vector.reserve(exampleCellSize);
            
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::uniform_real_distribution<> dist1((double) 1,(double) max);
            std::uniform_real_distribution<> dist2((double) min,(double) max);

            for (size_t i = 0; i < exampleCellSize; i++){
                T1 rand1 = (T1) dist1(rnd);
                T2 rand2 = (T2) dist2(rnd);
                
                keys_vector.push_back(rand1);
                values_vector.push_back(rand2);
            }

            std::copy(keys_vector.begin(),keys_vector.end(),exampleKeyList.begin());
            std::copy(values_vector.begin(),values_vector.end(),exampleValueList.begin());
        };

        //Mische verschiedene Schlüssel
        void shuffleKeys(){
            std::vector<T1> keys_vector;
            keys_vector.reserve(exampleCellSize);

            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::copy(exampleKeyList.begin(), exampleKeyList.end(),keys_vector.begin());
            std::shuffle(keys_vector.begin(), keys_vector.end(), rnd);

            std::copy(keys_vector.begin(),keys_vector.end(),exampleKeyList.begin());
        };

        //Drucke Zeitmessung für den Kernel
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
                std::cout << "Anzahl der gesuchten Zellen                 : ";
                std::cout << averageSumFound << std::endl;
                std::cout << std::endl;

            }else{
                size_t averageDeletedCells = benchmark_kernel.sum_num_deleted_cells/Test_Num;
                std::cout << "Löschung von Schlüsseln in der Hashtabelle bei ";
                std::cout << Test_Num << " Versuchen" << std::endl;
                std::cout << std::endl;
                std::cout << "Anzahl der gelöschten Zellen                : ";
                std::cout << averageDeletedCells << std::endl;
                std::cout << std::endl;
            }

            float averageDurationUpload, averageDurationRun, averageDurationDownload, averageDurationTotal;

            averageDurationUpload = benchmark_kernel.upload/Test_Num;
            averageDurationRun = benchmark_kernel.run/Test_Num;
            averageDurationDownload = benchmark_kernel.download/Test_Num;
            averageDurationTotal = benchmark_kernel.total/Test_Num;

            std::cout << "Dauer zum Hochladen                         : ";
            std::cout <<  averageDurationUpload *1000000 << std::endl;
            std::cout << "Dauer zur Ausführung                        : ";
            std::cout <<  averageDurationRun *1000000 << std::endl;
            std::cout << "Dauer zum Herunterladen                     : ";
            std::cout <<  averageDurationDownload *1000000 << std::endl;
            std::cout << "Gesamtdauer                                 : ";
            std::cout <<  averageDurationTotal *1000000 << std::endl;
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
            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

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
            
            Hash_Table<T1,T2> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            timer.stop();

            //hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Gesamtdauer                                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.insert_List(exampleKeyList.data(),exampleValueList.data(),exampleCellSize);
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
        
            //hash_table2.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten 
        //bei Test_Num Versuchen, d.h 100 Versuchen hinzu       
        void insertTestCells2(hash_type HashType){
            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

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

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                hash_table2.insert_List(exampleKeyList.data(),exampleValueList.data(),exampleCellSize);
                Benchmark Benchmark_Insert = hash_table2.getBenchmark(insert_hash_table);
           
                benchmark_kernel.upload+=Benchmark_Insert.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Insert.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Insert.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Insert.getDurationTotal();
            }

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

            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

            Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

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
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
            
            createCells();
            shuffleKeys();
            keyListArray = exampleKeyList.data();
            valueListArray = exampleValueList.data(); 

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) if (hash_table.search(keyListArray[i]) == true) ++sum_found;
            timer.stop();
            //hash_table1.print();
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
            hash_table.search_List(exampleKeyList.data(),exampleCellSize);
            Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);
            Benchmark_Search.print();
            //hash_table2.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Suche nach einer Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen   
        void searchTestCells2(hash_type HashType){
            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

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

            for (int i = 0; i < Test_Num; i++){
                Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
                createCells();
                shuffleKeys();

                hash_table.search_List(exampleKeyList.data(),exampleCellSize);

                Benchmark Benchmark_Search = hash_table.getBenchmark(search_hash_table);
                benchmark_kernel.upload+=Benchmark_Search.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Search.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Search.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Search.getDurationTotal();
                
                benchmark_kernel.SumFound+=Benchmark_Search.getSumFound();
            }

            printTotalBenchmark();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Sequentielle und parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle bei einem Versuch
        void deleteTestCells1(hash_type HashType, int min=0, int max=100){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t num_cells_prev1, num_cells_curr1;
            size_t num_cells_prev2, num_cells_curr2;

            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

            Hash_Table<T1,T2> hash_table1(HashType,examplefunction1,examplefunction2,exampleHashTableSize);

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
            
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            num_cells_prev1 = hash_table1.getNumCell();
            createCells(min,max);
            shuffleKeys();
            keyListArray = exampleKeyList.data();
            valueListArray = exampleValueList.data(); 

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<exampleCellSize; i++) hash_table1.deleteKey(keyListArray[i]);
            timer.stop();

            //hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Gesamtdauer                                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
            for (size_t i=0; i<exampleCellSize; i++) hash_table2.insert(keyListArray[i],valueListArray[i]);
            num_cells_prev2 = hash_table2.getNumCell();

            createCells(min,max);
            shuffleKeys();
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.delete_List(exampleKeyList.data(),exampleCellSize);
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

            //hash_table2.print();
        };

        /////////////////////////////////////////////////////////////////////////////////////////
        //Nur parallele Ausführung
        /////////////////////////////////////////////////////////////////////////////////////////
        //Lösche eine Liste von Schlüsseln in der Hashtabelle
        //bei Test_Num Versuchen, d.h 100 Versuchen    
        void deleteTestCells2(hash_type HashType, int min=0, int max=100){
            T1 * keyListArray = exampleKeyList.data();
            T2 * valueListArray = exampleValueList.data(); 

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

            for (int i = 0; i < Test_Num; i++){
                size_t num_cells_prev, num_cells_curr;

                Hash_Table<T1,T2> hash_table(HashType,examplefunction1,examplefunction2,exampleHashTableSize);
                for (size_t i=0; i<exampleCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
                num_cells_prev = hash_table.getNumCell();

                createCells(min,max);
                shuffleKeys();
                
                hash_table.delete_List(exampleKeyList.data(),exampleCellSize);
                Benchmark Benchmark_Delete = hash_table.getBenchmark(delete_hash_table);
                num_cells_curr = num_cells_prev - hash_table.getNumCell();

                benchmark_kernel.upload+=Benchmark_Delete.getDurationUpload();
                benchmark_kernel.run+=Benchmark_Delete.getDurationRun(); 
                benchmark_kernel.download+=Benchmark_Delete.getDurationDownload();
                benchmark_kernel.total+=Benchmark_Delete.getDurationTotal();
                
                benchmark_kernel.sum_num_deleted_cells+=num_cells_curr;
            }

            printTotalBenchmark();
        };
};


#endif