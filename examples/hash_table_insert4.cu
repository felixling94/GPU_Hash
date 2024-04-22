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
//Laufzeitvergleich von einer gegegeben Datei zwischen verschiedenen Hashverfahren bei
//a. einer gegebenen 1. und 2. Hashfunktionen, 
//b. einer gegebenen Anzahl von Schlüsseln, 
//c. unterschiedlichen oder gleichen Schlüsselgrößen, und
//d. einem gegebenen Auslastungsgrad von einer oder zwei Hashtabellen
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    char* fileName;
    size_t exampleHashTableSize, exampleKeyNum, matrix_size;
    double occupancy;
    int function_code1, function_code2;
    hash_function hash_function1, hash_function2; 

    int deviceID{0};
    struct cudaDeviceProp props;

    if(argc < 6){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    fileName = argv[1];
    exampleKeyNum = (size_t) atoi(argv[2]);
    occupancy = atof(argv[3]);
    function_code1 = atoi(argv[4]);
    function_code2 = atoi(argv[5]);

    if (exampleKeyNum <=0){
        std::cout << "Die Anzahl an Schlüssel muss mehr als Null betragen." << std::endl;
        return -1;
    }

    if (occupancy <=0){
        std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
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

    matrix_size = exampleKeyNum * sizeof(uint32_t);
    exampleHashTableSize = (size_t) ceil((double) (exampleKeyNum) / occupancy);

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "GPU" << "," << props.name << std::endl;
    std::cout << "VRAM" << "," << (props.totalGlobalMem/1024)/1024 << "MB" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten" << ",";
    std::cout << ((matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "MB\n" << std::endl;
    std::cout << std::endl;
    std::cout << "Anzahl der gespeicherten Zellen" << "," << exampleKeyNum << std::endl;
    std::cout << "Größe der Hashtabelle" << "," << exampleHashTableSize << std::endl;
    std::cout << "Größe der Cuckoo-Hashtabelle" << "," << 2*exampleHashTableSize << std::endl;

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

    Example_Hash_Table<uint32_t> example_hash_table(exampleKeyNum,exampleHashTableSize,hash_function1,hash_function2);
    example_hash_table.readCells(fileName);

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
    std::cout << "Gesamtdauer" << "," <<  timer.getDuration() << std::endl;
    
    return 0;
};