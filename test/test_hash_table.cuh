#ifndef TEST_HASH_TABLE_CUH
#define TEST_HASH_TABLE_CUH

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <../include/base.h>
#include <../include/hash_table.h>
#include <../core/hash_table.cuh>
#include <../tools/timer.cuh>

template <typename T1, typename T2>
class Test_Hash_Table{
    private:
        std::vector<T1> testKeyList;
        std::vector<T2> testValueList;

        size_t testCellSize = 11;
        size_t testHashTableSize = testCellSize + 5;

        hash_function testfunction1 = modulo;
        hash_function testfunction2;

    public:
        //Konstruktor
        Test_Hash_Table(){
            testKeyList.reserve(testCellSize);
            testValueList.reserve(testCellSize);
        };
        
        Test_Hash_Table(size_t test_cell_size, size_t test_table_size,hash_function function1, hash_function function2=modulo):
        testCellSize(test_cell_size),testHashTableSize(test_table_size),testfunction1(function1),testfunction2(function2){
            if (test_cell_size>test_table_size){
                std::cout << "Die Hashtabelle darf höchstens maximal " << test_table_size;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << test_cell_size<< "." << std::endl;
                exit (EXIT_FAILURE);
            }
            
            testKeyList.reserve(test_cell_size);
            testValueList.reserve(test_cell_size);
        };

        //Destruktor
        ~Test_Hash_Table(){};

        //Gebe Schlüssel zurück
        std::vector<T1> getKeys(){
            return testKeyList;
        };

        //Gebe Werte zurück
        std::vector<T2> getKValues(){
            return testValueList;
        };

        //Erzeuge verschiedene Werte für die Schlüssel und Werte zufällig
        void createCells(int min=0, int max=100){
            std::vector<T1> keys_vector;
            std::vector<T2> values_vector;
            
            keys_vector.reserve(testCellSize);
            values_vector.reserve(testCellSize);
            
            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::uniform_int_distribution<T1> dist1(1,max);
            std::uniform_int_distribution<T2> dist2(min,max);

            for (size_t i = 0; i < testCellSize; i++){
                T1 rand1 = dist1(rnd);
                T2 rand2 = dist2(rnd);

                keys_vector.push_back(rand1);
                values_vector.push_back(rand2);
            }

            std::copy(keys_vector.begin(),keys_vector.end(),testKeyList.begin());
            std::copy(values_vector.begin(),values_vector.end(),testValueList.begin());
        };

        //Mische verschiedene Schlüssel
        void shuffleKeys(){
            std::vector<T1> keys_vector;
            keys_vector.reserve(testCellSize);

            std::random_device generator;
            size_t seed = generator();
            std::mt19937 rnd(seed);
            
            std::copy(testKeyList.begin(), testKeyList.end(),keys_vector.begin());
            std::shuffle(keys_vector.begin(), keys_vector.end(), rnd);

            std::copy(keys_vector.begin(),keys_vector.end(),testKeyList.begin());
        };

        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten hinzu       
        void insertTestCells(hash_type HashType){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            T1 * keyListArray = testKeyList.data();
            T2 * valueListArray = testValueList.data(); 

            Hash_Table<T1,T2> hash_table1(HashType,testfunction1,testfunction2,testHashTableSize);

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
            
            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<testCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            timer.stop();
            //hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Anzahl der Zellen in der Hashtabelle        : ";
            std::cout << hash_table1.getNumCell() << std::endl;
            std::cout << "Gesamtdauer (Millisekunden)                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,testfunction1,testfunction2,testHashTableSize);
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.insert_List(testKeyList.data(),testValueList.data(),testCellSize);
            //hash_table2.print();
        };

        //Suche nach einer Liste von Schlüsseln in der Hashtabelle
        void searchTestCells(hash_type HashType){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t sum_found = 0;

            T1 * keyListArray = testKeyList.data();
            T2 * valueListArray = testValueList.data(); 

            Hash_Table<T1,T2> hash_table(HashType,testfunction1,testfunction2,testHashTableSize);

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
            
            for (size_t i=0; i<testCellSize; i++) hash_table.insert(keyListArray[i],valueListArray[i]);
            createCells();
            shuffleKeys();

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<testCellSize; i++) if (hash_table.search(keyListArray[i]) == true) ++sum_found;
            timer.stop();
            //hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Anzahl der gesuchten Zellen                 : ";
            std::cout << sum_found << std::endl;
            std::cout << "Gesamtdauer (Millisekunden)                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table.search_List(testKeyList.data(),testCellSize);
            //hash_table2.print();
        };

        //Lösche eine Liste von Schlüsseln in der Hashtabelle
        void deleteTestCells(hash_type HashType){
            //1. Deklariere und initialisiere alle Variablen
            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            size_t num_cells_prev, num_cells_curr;

            T1 * keyListArray = testKeyList.data();
            T2 * valueListArray = testValueList.data(); 

            Hash_Table<T1,T2> hash_table1(HashType,testfunction1,testfunction2,testHashTableSize);

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
            
            for (size_t i=0; i<testCellSize; i++) hash_table1.insert(keyListArray[i],valueListArray[i]);
            num_cells_prev = hash_table1.getNumCell();
            createCells();
            shuffleKeys();

            CPUTimer timer;
            timer.start();
            for (size_t i=0; i<testCellSize; i++) hash_table1.deleteKey(keyListArray[i]);
            timer.stop();

            num_cells_curr = num_cells_prev - hash_table1.getNumCell();

            //hash_table1.print();
            //Fasse Resultate für jede Hashverfahren zusammen
            std::cout << "Anzahl der gelöschten Zellen                : ";
            std::cout << num_cells_curr << std::endl;
            std::cout << "Gesamtdauer (Millisekunden)                 : ";
            std::cout << timer.getDuration() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hash_Table<T1,T2> hash_table2(HashType,testfunction1,testfunction2,testHashTableSize);
            for (size_t i=0; i<testCellSize; i++) hash_table2.insert(keyListArray[i],valueListArray[i]);
            createCells();
            shuffleKeys();
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hash_table2.delete_List(testKeyList.data(),testCellSize);
            //hash_table2.print();
        };
};


#endif