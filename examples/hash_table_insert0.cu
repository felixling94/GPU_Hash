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
//b. einer gegebenen Anzahl von Schlüsseln, und
//c. einem gegebenen Auslastungsgrad von einer oder zwei Hashtabellen
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    int exampleKeyMinSize, exampleKeyMaxSize;
    
    size_t exampleHashTableSize, exampleKeyNum;
    double occupancy;
    int function_code1, function_code2;
    hash_function hash_function1, hash_function2; 

    int deviceID{0};
    struct cudaDeviceProp props;

    if(argc < 5){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    exampleKeyNum = (size_t) atoi(argv[1]);
    occupancy = atof(argv[2]);
    function_code1 = atoi(argv[3]);
    function_code2 = atoi(argv[4]);

    if (exampleKeyNum <=0){
        std::cout << "Die Anzahl an Schlüssel muss mehr als Null betragen." << std::endl;
        return -1;
    }

    if (occupancy <=0){
        std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
        return -1;
    }
    
    if (function_code1<1 || function_code1>11){
        std::cout << "Der Kode einer 1. Hashfunktion muss innerhalb des Bereiches von 1 bis 11 sein." << std::endl;
        return -1;
    }

    if (function_code2<1 || function_code2>11){
        std::cout << "Der Kode einer 2. Hashfunktion muss innerhalb des Bereiches von 1 bis 11 sein." << std::endl;
        return -1;
    }

    const size_t matrix_size{exampleKeyNum * sizeof(uint32_t)};
    exampleKeyMinSize = 0;
    exampleKeyMaxSize = (int) exampleKeyNum;

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
    std::cout << "Ausgewähltes " << props.name << " mit "
              << (props.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << (( matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "mb\n" << std::endl;

    exampleHashTableSize = (size_t) ceil((double) (exampleKeyNum) / occupancy);

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;   
    std::cout << "Anzahl der gespeicherten Zellen             : ";
    std::cout << exampleKeyNum << std::endl;
    std::cout << "Größe der Hashtabelle                       : ";
    std::cout << exampleHashTableSize << std::endl;
    std::cout << "Größe der Cuckoo-Hashtabelle                : ";
    std::cout << 2*exampleHashTableSize << std::endl;

    std::cout << std::endl;
    if (function_code1 == 2){
        hash_function1 = multiplication;
        std::cout << "1. Hashfunktion: Multiplikative Methode" << std::endl;
    }else if (function_code1 == 3){
        hash_function1 = murmer;
        std::cout << "1. Hashfunktion: Murmer Hash" << std::endl;
    }else if (function_code1 == 4){
        hash_function1 = universal0;
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 290000  b: 320000  Primzahl: 320114)" << std::endl;
    }else if (function_code1 == 5){
        hash_function1 = universal1;
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 149400  b: 149500  Primzahl: 149969)" << std::endl;
    }else if (function_code1 == 6){
        hash_function1 = universal2;
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 135  b: 140  Primzahl: 149)" << std::endl;
    }else if (function_code1 == 7){
        hash_function1 = dycuckoo_hash1;
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 1" << std::endl;
    }else if (function_code1 == 8){
        hash_function1 = dycuckoo_hash2;
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 2" << std::endl;
    }else if (function_code1 == 9){
        hash_function1 = dycuckoo_hash3;
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 3" << std::endl;
    }else if (function_code1 == 10){
        hash_function1 = dycuckoo_hash4;
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 4" << std::endl;
    }else if (function_code1 == 11) {
        hash_function1 = dycuckoo_hash5;
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 5" << std::endl;
    }else{
        hash_function1 = modulo;
        std::cout << "1. Hashfunktion: Divisions-Rest-Methode" << std::endl;
    }

    if (function_code2 == 2){
        hash_function2 = multiplication;
        std::cout << "2. Hashfunktion: Multiplikative Methode" << std::endl;
    }else if (function_code2 == 3){
        hash_function2 = murmer;
        std::cout << "2. Hashfunktion: Murmer Hash" << std::endl;
    }else if (function_code2 == 4){
        hash_function2 = universal0;
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 290000  b: 320000  Primzahl: 320114)" << std::endl;
    }else if (function_code2 == 5){
        hash_function2 = universal1;
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 149400  b: 149500  Primzahl: 149969)" << std::endl;
    }else if (function_code2 == 6){
        hash_function2 = universal2;
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 135  b: 140  Primzahl: 149)" << std::endl;
    }else if (function_code2 == 7){
        hash_function2 = dycuckoo_hash1;
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 1" << std::endl;
    }else if (function_code2 == 8){
        hash_function2 = dycuckoo_hash2;
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 2" << std::endl;
    }else if (function_code2 == 9){
        hash_function2 = dycuckoo_hash3;
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 3" << std::endl;
    }else if (function_code2 == 10){
        hash_function2 = dycuckoo_hash4;
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 4" << std::endl;
    }else if (function_code2 == 11) {
        hash_function2 = dycuckoo_hash5;
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 5" << std::endl;
    }else{
        hash_function2 = modulo;
        std::cout << "2. Hashfunktion: Divisions-Rest-Methode" << std::endl;
    }
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table(exampleKeyNum,exampleHashTableSize,hash_function1,hash_function2);
    example_hash_table.createCells(exampleKeyMinSize,exampleKeyMaxSize);

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Keine Kollionsauflösung
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells1(no_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells1(linear_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells1(quadratic_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Doppelte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells1(double_probe);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Cuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    example_hash_table.insertTestCells1(cuckoo_probe);
    /////////////////////////////////////////////////////////////////////////////////////////

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer für alle offenen Hashverfahren  : ";
    std::cout << timer.getDuration() << std::endl;
    std::cout << "(in Millisekunden)" << std::endl;
    
    return 0;
};