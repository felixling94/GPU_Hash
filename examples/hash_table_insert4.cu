#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdint.h>

#include "example_hash_table.cuh"
#include <../include/base.h>
#include <../tools/timer.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Laufzeitvergleich zwischen verschiedener Anzahl von Schlüsseln bei  
//a. einer gegebenen 1. und 2. Hashfunktionen, und
//b. gegebenen Hashverfahren, z.B. linearem Sondieren
/////////////////////////////////////////////////////////////////////////////////////////

//Führe Hashverfahren mit verschiedenen Datentypen aus
template <typename T1, typename T2>
void runMain(hash_type type, hash_function function1, hash_function function2, size_t keyNum){
    int deviceID{0};
    struct cudaDeviceProp props;
    const size_t matrix_size{keyNum * sizeof(T1)};
    const size_t hashTableSize{(size_t) (keyNum*120/100)};

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);
    
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
    std::cout << "Ausgewähltes " << props.name << " mit "
              << (props.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << (( matrix_size * 3 + sizeof(T1)) / 1024 / 1024) << "mb\n" << std::endl;
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;

    if (function1 == multiplication){
        std::cout << "1. Hashfunktion: Multiplikative Methode" << std::endl;
    }else if (function1 == murmer){
        std::cout << "1. Hashfunktion: Murmer Hash" << std::endl;
    }else if (function1 == universal0){
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 290000  b: 320000  Primzahl: 320114)" << std::endl;
    }else if (function1 == universal1){
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 149400  b: 149500  Primzahl: 149969)" << std::endl;
    }else if (function1 == universal2){
        std::cout << "1. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 135  b: 140  Primzahl: 149)" << std::endl;
    }else if (function1 == dycuckoo_hash1){
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 1" << std::endl;
    }else if (function1 == dycuckoo_hash2){
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 2" << std::endl;
    }else if (function1 == dycuckoo_hash3){
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 3" << std::endl;
    }else if (function1 == dycuckoo_hash4){
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 4" << std::endl;
    }else if (function1 == dycuckoo_hash5) {
        std::cout << "1. Hashfunktion: DyCuckoo-Hash 5" << std::endl;
    }else{
        std::cout << "1. Hashfunktion: Divisions-Rest-Methode" << std::endl;
    }

    if (function2 == multiplication){
        std::cout << "2. Hashfunktion: Multiplikative Methode" << std::endl;
    }else if (function2 == murmer){
        std::cout << "2. Hashfunktion: Murmer Hash" << std::endl;
    }else if (function2 == universal0){
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 290000  b: 320000  Primzahl: 320114)" << std::endl;
    }else if (function2 == universal1){
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 149400  b: 149500  Primzahl: 149969)" << std::endl;
    }else if (function2 == universal2){
        std::cout << "2. Hashfunktion: Universelle Hashfunktion" << std::endl;
        std::cout << "                 (a: 135  b: 140  Primzahl: 149)" << std::endl;
    }else if (function2 == dycuckoo_hash1){
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 1" << std::endl;
    }else if (function2 == dycuckoo_hash2){
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 2" << std::endl;
    }else if (function2 == dycuckoo_hash3){
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 3" << std::endl;
    }else if (function2 == dycuckoo_hash4){
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 4" << std::endl;
    }else if (function2 == dycuckoo_hash5) {
        std::cout << "2. Hashfunktion: DyCuckoo-Hash 5" << std::endl;
    }else{
        std::cout << "2. Hashfunktion: Divisions-Rest-Methode" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Anzahl der gespeicherten Zellen             : ";
    std::cout << keyNum << std::endl;
    if (type != cuckoo_probe){
        std::cout << "Größe der Hashtabelle                       : ";
        std::cout << hashTableSize << std::endl;
    }else{
        std::cout << "Größe der Cuckoo-Hashtabelle                : ";
        std::cout << 2*hashTableSize << std::endl;
    }
    std::cout << std::endl;

    Example_Hash_Table<T1,T2> example_hash_table(keyNum,hashTableSize,function1,function2);
    example_hash_table.createCells(1,(int)keyNum*2);
    example_hash_table.insertTestCells2(type);
};

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    const size_t * key_num = new size_t[5]{320114,25000,200000,140000,100019};
    size_t * exampleHashTableSize = new size_t[5];
    int function_code1, function_code2, hash_type_code;
    hash_function hash_function1, hash_function2;
    hash_type hash_type1; 
    
    if(argc < 4){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    hash_type_code = atoi(argv[1]);
    function_code1 = atoi(argv[2]);
    function_code2 = atoi(argv[3]);

    if (hash_type_code<0 || hash_type_code>3){
        std::cout << "Der Kode eines Hashtyps muss innerhalb des Bereiches von 0 bis 3 sein." << std::endl;
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
    
    if (hash_type_code == 1){
        hash_type1 = quadratic_probe;
    }else if(hash_type_code == 2){
        hash_type1 = double_probe;
    }else if(hash_type_code == 3){
        hash_type1 = cuckoo_probe;
    }else{
        hash_type1 = linear_probe;
    }

    if (function_code1 == 2){
        hash_function1 = multiplication;
    }else if (function_code1 == 3){
        hash_function1 = murmer;
    }else if (function_code1 == 4){
        hash_function1 = universal0;
    }else if (function_code1 == 5){
        hash_function1 = universal1;
    }else if (function_code1 == 6){
        hash_function1 = universal2;
    }else if (function_code1 == 7){
        hash_function1 = dycuckoo_hash1;
    }else if (function_code1 == 8){
        hash_function1 = dycuckoo_hash2;
    }else if (function_code1 == 9){
        hash_function1 = dycuckoo_hash3;
    }else if (function_code1 == 10){
        hash_function1 = dycuckoo_hash4;
    }else if (function_code1 == 11) {
        hash_function1 = dycuckoo_hash5;
    }else{
        hash_function1 = modulo;
    }

    if (function_code2 == 2){
        hash_function2 = multiplication;
    }else if (function_code2 == 3){
        hash_function2 = murmer;
    }else if (function_code2 == 4){
        hash_function2 = universal0;
    }else if (function_code2 == 5){
        hash_function2 = universal1;
    }else if (function_code2 == 6){
        hash_function2 = universal2;
    }else if (function_code2 == 7){
        hash_function2 = dycuckoo_hash1;
    }else if (function_code2 == 8){
        hash_function2 = dycuckoo_hash2;
    }else if (function_code2 == 9){
        hash_function2 = dycuckoo_hash3;
    }else if (function_code2 == 10){
        hash_function2 = dycuckoo_hash4;
    }else if (function_code2 == 11) {
        hash_function2 = dycuckoo_hash5;
    }else{
        hash_function2 = modulo;
    }

    CPUTimer timer;
    timer.start();

    for (size_t i = 0; i<5; i++) runMain<uint32_t,uint32_t>(hash_type1, hash_function1, hash_function2, key_num[i]);
  
    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer                                 : ";
    std::cout << timer.getDuration() << std::endl;
    
    return 0;
};