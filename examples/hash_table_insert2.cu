#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <stdint.h>

#include "example_hash_table.cuh"
#include <../include/base.h>
#include <../tools/timer.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Laufzeitvergleich zwischen verschiedenen Hashverfahren bei
//a. einer gegebenen 1. und 2. Hashfunktionen, 
//b. einer gegebenen Anzahl von Schlüsseln, 
//c. unterschiedlichen oder gleichen Schlüsselgrößen, und
//d. einem gegebenen Auslastungsgrad von einer oder zwei Hashtabellen
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    size_t exampleHashTableSize, exampleKeyNum, exampleOccupyKeyNum, matrix_size;
    int exampleBlockNum, exampleThreadsPerBlock;
    double occupancy;
    int function_code1, function_code2, int_value_same;
    bool value_same;
    hash_function hash_function1, hash_function2; 

    int deviceID{0};
    struct cudaDeviceProp props;

    if(argc < 9){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }
    
    int_value_same = atoi(argv[1]);
    exampleKeyNum = (size_t) atoi(argv[2]);
    exampleHashTableSize = (size_t) atoi(argv[3]); 
    occupancy = atof(argv[4]);
    function_code1 = atoi(argv[5]);
    function_code2 = atoi(argv[6]);
    exampleBlockNum = atoi(argv[7]);
    exampleThreadsPerBlock= atoi(argv[8]);
    
    if (int_value_same<0 || int_value_same>1){
        std::cout << "Der Code der Gleichheit der Schlüsselgröße muss entweder 0 bis 1 sein." << std::endl;
        return -1;
    }

    if (exampleHashTableSize < 1){
        std::cout << "Die Größe einer oder zwei Hashtabelle(n) muss mehr als Null betragen." << std::endl;
        return -1;
    }

    if (occupancy < 0){
        std::cout << "Der Auslastungsfaktor der Hashtabelle muss mindestens Null betragen." << std::endl;
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
    
    if (exampleBlockNum < 1 || exampleThreadsPerBlock < 1 || 
        exampleKeyNum != (size_t)(exampleBlockNum*exampleThreadsPerBlock)){
        exampleBlockNum = (int) exampleKeyNum;
        exampleThreadsPerBlock = 1;
    }

    matrix_size = exampleKeyNum * sizeof(cell<uint32_t,uint32_t>);
    exampleOccupyKeyNum = (size_t) ceil((double) (exampleHashTableSize) * occupancy);

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "GPU" << "," << props.name << std::endl;
    std::cout << "VRAM" << "," << (props.totalGlobalMem/1024)/1024 << "MB" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten" << ",";
    std::cout << ((matrix_size * 3 + sizeof(cell<uint32_t,uint32_t>)) / 1024 / 1024) << "MB\n" << std::endl;
    std::cout << "Block_Zahl" << "," << "Threads_Zahl_Pro_Block" << std::endl;
    std::cout << exampleBlockNum << "," << exampleThreadsPerBlock << std::endl;
    std::cout << std::endl;   
    std::cout << "Anzahl der gespeicherten Zellen" << "," << exampleKeyNum << std::endl;
    std::cout << "Größe der Hashtabelle" << "," << exampleHashTableSize << std::endl;
    std::cout << "Größe der Cuckoo-Hashtabellen" << "," << 2*exampleHashTableSize << std::endl;
    
    if (int_value_same == 1){
        value_same = true;
    }else{
        value_same = false;     
    }

    std::cout << std::endl;
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

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table(exampleKeyNum,exampleOccupyKeyNum,exampleHashTableSize,hash_function1,hash_function2,
                                                             exampleBlockNum,exampleThreadsPerBlock);
    example_hash_table.createCells(value_same);

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Keine Kollionsauflösung
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells2(no_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells2(linear_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells2(quadratic_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Doppelte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells2(double_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Cuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells2(cuckoo_probe);
    /////////////////////////////////////////////////////////////////////////////////////////

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
    
    return 0;
};