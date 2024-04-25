#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdint.h>

#include "example_hash_table.cuh"
#include <../include/base.h>
#include <../tools/timer.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Laufzeitvergleich von einer Datei zwischen verschiedenen Auslastungsgraden einer Hashtabelle bei
//a. einer gegebenen Anzahl von Schlüsseln, 
//b. gleichen oder unterschiedlichen Schlüsselgrößen, 
//c. einer gegebenen 1. und 2. Hashfunktionen, und
//d. gegebenen Hashverfahren, z.B. linearem Sondieren
/////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void runKernel(int block_num, int num_threads_per_block){
    int deviceID{0};
    struct cudaDeviceProp props;
    const size_t matrix_size{block_num*num_threads_per_block * sizeof(T)};

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);
    
    std::cout << "GPU" << "," << props.name << std::endl;
    std::cout << "VRAM" << "," << (props.totalGlobalMem/1024)/1024 << "MB" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten" << ",";
    std::cout << ((matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "MB\n" << std::endl;
    std::cout << "Block_Zahl" << "," << "Threads_Zahl_Pro_Block" << std::endl;
    std::cout << block_num << "," << num_threads_per_block << std::endl;
    std::cout << std::endl;     
};

//Führe Hashverfahren mit verschiedenen Datentypen aus
template <typename T>
void runMain(hash_type type, hash_function function1, hash_function function2, int block_num, int num_threads_per_block, double occupancy, char* fileName){
    const size_t hashTableSize{(size_t) ceil((double) (block_num * num_threads_per_block) / occupancy)};
   
    std::cout << "Anzahl der gespeicherten Zellen" << "," << block_num*num_threads_per_block << std::endl;
    if (type != cuckoo_probe){
        std::cout << "Größe der Hashtabelle" << "," << hashTableSize << std::endl;
    }else{
        std::cout << "Größe der Cuckoo-Hashtabelle" << "," << 2*hashTableSize << std::endl;
    }
    std::cout << "Auslastungsfaktor der Hashtabelle" << "," << occupancy << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<T> example_hash_table(block_num, num_threads_per_block,hashTableSize,function1,function2);
    example_hash_table.readCells(fileName);
    example_hash_table.insertTestCells2(type);
};

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    char* fileName;
    int exampleBlockNum, exampleThreadsPerBlock;
    const double * occupancy = new double[5]{1.0,0.8,0.6,0.4,0.2};

    size_t * exampleHashTableSize = new size_t[5];
    int function_code1, function_code2;
    hash_function hash_function1, hash_function2;
    
    if(argc < 6){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    fileName = argv[1];
    exampleBlockNum = atoi(argv[2]);
    exampleThreadsPerBlock= atoi(argv[3]);
    function_code1 = atoi(argv[4]);
    function_code2 = atoi(argv[5]);

    if (exampleBlockNum <=0){
        std::cout << "Die Anzahl an Blöcke muss mehr als Null betragen." << std::endl;
        return -1;
    }

    if (exampleThreadsPerBlock <=0){
        std::cout << "Die Anzahl an Threads pro Block muss mehr als Null betragen." << std::endl;
        return -1;
    }

    if (function_code1<1 || function_code1>11){
        std::cout << "Der Code einer 1. Hashfunktion muss innerhalb des Bereiches von 1 bis 11 sein." << std::endl;
        return -1;
    }

    if (function_code2<1 || function_code2>11){
        std::cout << "Der Code einer 2. Hashfunktion muss innerhalb des Bereiches von 1 bis 11 sein." << std::endl;
        return -1;
    }

    runKernel<uint32_t>(exampleBlockNum, exampleThreadsPerBlock);

    if (function_code1 == 2){
        hash_function1 = multiplication;
        std::cout << "1. Hashfunktion" << "," << "Multiplikative Methode" << std::endl;
    }else if (function_code1 == 3){
        hash_function1 = murmer;
        std::cout << "1. Hashfunktion" << "," << "Murmer-Hashfunktion" << std::endl;
    }else if (function_code1 == 4){
        hash_function1 = universal0;
        std::cout << "1. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 20019" << "," << "b: 20025" << "," <<  "Primzahl: 20029" << std::endl;
        std::cout << std::endl;
    }else if (function_code1 == 5){
        hash_function1 = universal1;
        std::cout << "1. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 10023" << "," << "b: 10037" << "," <<  "Primzahl: 10039" << std::endl;
        std::cout << std::endl;
    }else if (function_code1 == 6){
        hash_function1 = universal2;
        std::cout << "1. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 5029" << "," << "b: 5038" << "," <<  "Primzahl: 5039" << std::endl;
        std::cout << std::endl;
    }else if (function_code1 == 7){
        hash_function1 = dycuckoo_hash1;
        std::cout << "1. Hashfunktion" << "," << "DyCuckoo-1" << std::endl;
    }else if (function_code1 == 8){
        hash_function1 = dycuckoo_hash2;
        std::cout << "1. Hashfunktion" << "," << "DyCuckoo-2" << std::endl;
    }else if (function_code1 == 9){
        hash_function1 = dycuckoo_hash3;
        std::cout << "1. Hashfunktion" << "," << "DyCuckoo-3" << std::endl;
    }else if (function_code1 == 10){
        hash_function1 = dycuckoo_hash4;
        std::cout << "1. Hashfunktion" << "," << "DyCuckoo-4" << std::endl;
    }else if (function_code1 == 11) {
        hash_function1 = dycuckoo_hash5;
        std::cout << "1. Hashfunktion" << "," << "DyCuckoo-5" << std::endl;
    }else{
        hash_function1 = modulo;
        std::cout << "1. Hashfunktion" << "," << "Divisions-Rest-Methode" << std::endl;
    }

    if (function_code2 == 2){
        hash_function2 = multiplication;
        std::cout << "2. Hashfunktion" << "," << "Multiplikative Methode" << std::endl;
    }else if (function_code2 == 3){
        hash_function2 = murmer;
        std::cout << "2. Hashfunktion" << "," << "Murmer-Hashfunktion" << std::endl;
    }else if (function_code2 == 4){
        hash_function2 = universal0;
        std::cout << "2. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 20019" << "," << "b: 20025" << "," <<  "Primzahl: 20029" << std::endl;
        std::cout << std::endl;
    }else if (function_code2 == 5){
        hash_function2 = universal1;
        std::cout << "2. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 10023" << "," << "b: 10037" << "," <<  "Primzahl: 10039" << std::endl;
        std::cout << std::endl;
    }else if (function_code2 == 6){
        hash_function2 = universal2;
        std::cout << "2. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
        std::cout << "," << "a: 5029" << "," << "b: 5038" << "," <<  "Primzahl: 5039" << std::endl;
        std::cout << std::endl;
    }else if (function_code2 == 7){
        hash_function2 = dycuckoo_hash1;
        std::cout << "2. Hashfunktion" << "," << "DyCuckoo-1" << std::endl;
    }else if (function_code2 == 8){
        hash_function2 = dycuckoo_hash2;
        std::cout << "2. Hashfunktion" << "," << "DyCuckoo-2" << std::endl;
    }else if (function_code2 == 9){
        hash_function2 = dycuckoo_hash3;
        std::cout << "2. Hashfunktion" << "," << "DyCuckoo-3" << std::endl;
    }else if (function_code2 == 10){
        hash_function2 = dycuckoo_hash4;
        std::cout << "2. Hashfunktion" << "," << "DyCuckoo-4" << std::endl;
    }else if (function_code2 == 11) {
        hash_function2 = dycuckoo_hash5;
        std::cout << "2. Hashfunktion" << "," << "DyCuckoo-5" << std::endl;
    }else{
        hash_function2 = modulo;
        std::cout << "2. Hashfunktion" << "," << "Divisions-Rest-Methode" << std::endl;
    }
    std::cout << std::endl;
    
    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i<5; i++) runMain<uint32_t>(linear_probe, hash_function1, hash_function2, exampleBlockNum, exampleThreadsPerBlock, occupancy[i], fileName);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i<5; i++) runMain<uint32_t>(quadratic_probe, hash_function1, hash_function2, exampleBlockNum, exampleThreadsPerBlock, occupancy[i], fileName);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Doppelte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i<5; i++) runMain<uint32_t>(double_probe, hash_function1, hash_function2, exampleBlockNum, exampleThreadsPerBlock, occupancy[i], fileName);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Cuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i<5; i++) runMain<uint32_t>(cuckoo_probe, hash_function1, hash_function2, exampleBlockNum, exampleThreadsPerBlock, occupancy[i], fileName);
    /////////////////////////////////////////////////////////////////////////////////////////

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
    
    return 0;
};