#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdint.h>

#include "example_hash_table.cuh"
#include <../include/base.h>
#include <../tools/timer.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Laufzeitvergleich zwischen verschiedenen Auslastungsgraden einer Hashtabelle bei
//a. einer gegebenen Anzahl von Schlüsseln, 
//b. gleichen oder unterschiedlichen Schlüsselgrößen, 
//c. einer gegebenen 1. und 2. Hashfunktionen, und
//d. gegebenen Hashverfahren, z.B. linearem Sondieren
/////////////////////////////////////////////////////////////////////////////////////////

const size_t key_num{576};

template <typename T>
void runKernel(){
    int deviceID{0};
    struct cudaDeviceProp props;
    const size_t matrix_size{key_num * sizeof(T)};

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);
    
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
    std::cout << "Ausgewähltes " << props.name << " mit "
              << (props.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << (( matrix_size * 3 + sizeof(T)) / 1024 / 1024) << "mb\n" << std::endl;

};

//Führe Hashverfahren mit verschiedenen Datentypen aus
template <typename T>
void runMain(hash_type type, hash_function function1, hash_function function2, double occupancy, bool key_length_same){
    const size_t hashTableSize{(size_t) ceil((double) (key_num) / occupancy)};
   
    std::cout << "Anzahl der gespeicherten Zellen             : ";
    std::cout << key_num << std::endl;
    if (type != cuckoo_probe){
        std::cout << "Größe der Hashtabelle                       : ";
        std::cout << hashTableSize << std::endl;
    }else{
        std::cout << "Größe der Cuckoo-Hashtabelle                : ";
        std::cout << 2*hashTableSize << std::endl;
    }
    std::cout << "Auslastungsfaktor der Hashtabelle           : ";
    std::cout << occupancy << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<T> example_hash_table(key_num,hashTableSize,function1,function2);
    example_hash_table.createCells(key_length_same);
    example_hash_table.insertTestCells2(type);
    
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
};

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    const double * occupancy = new double[5]{1.0,0.8,0.6,0.4,0.2};
    size_t * exampleHashTableSize = new size_t[5];
    int function_code1, function_code2, hash_type_code, int_key_length_same;
    hash_function hash_function1, hash_function2;
    hash_type hash_type1;
    bool key_length_same; 
    
    if(argc < 5){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    int_key_length_same = atoi(argv[1]);
    hash_type_code = atoi(argv[2]);
    function_code1 = atoi(argv[3]);
    function_code2 = atoi(argv[4]);

    if (int_key_length_same<0 || int_key_length_same>1){
        std::cout << "Der Kode der Gleichheit der Schlüsselgröße muss entweder 0 bis 1 sein." << std::endl;
        return -1;
    }

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

    runKernel<uint32_t>();
    
    if (int_key_length_same == 1){
        key_length_same = true;
    }else{
        key_length_same = false;     
    }
      
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
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
    
    CPUTimer timer;
    timer.start();
    
    for (size_t i = 0; i<5; i++) runMain<uint32_t>(hash_type1, hash_function1, hash_function2, occupancy[i],key_length_same);
  
    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer                                 : ";
    std::cout << timer.getDuration() << std::endl;
    
    return 0;
};