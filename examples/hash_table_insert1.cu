#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <stdint.h>

#include "example_hash_table.cuh"
#include <../include/base.h>
#include <../tools/timer.cuh>

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    const size_t exampleKeyLength{800*200*200};
    const size_t exampleHashTableSize{exampleKeyLength};
    const size_t matrix_size{exampleKeyLength * sizeof(uint32_t)};

    int deviceID{0};
    struct cudaDeviceProp props;

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
    std::cout << "Ausgewähltes " << props.name << " mit "
              << (props.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << (( matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "mb\n" << std::endl;

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;   
    std::cout << "Anzahl der gespeicherten Zellen             : ";
    std::cout << exampleKeyLength << std::endl;
    std::cout << "Größe der Hashtabelle                       : ";
    std::cout << exampleHashTableSize << std::endl;

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Modulo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "1. Hashfunktion: Divisions-Rest-Methode" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table1(exampleKeyLength,exampleHashTableSize,modulo);
    example_hash_table1.createCells(1,(int)exampleKeyLength*2);
    example_hash_table1.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //Multiplikative Methode
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "2. Hashfunktion: Multiplikative Methode" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table2(exampleKeyLength,exampleHashTableSize,multiplication);
    example_hash_table2.createCells(1,(int)exampleKeyLength*2);
    example_hash_table2.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //Murmer-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "3. Hashfunktion: Murmer-Hashfunktion" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table3(exampleKeyLength,exampleHashTableSize,murmer);
    example_hash_table3.createCells(1,(int)exampleKeyLength*2);
    example_hash_table3.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //1. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "4. Hashfunktion: Universelle Hashfunktion" << std::endl;
    std::cout << "                 (a: 34999950  b: 34999960  Primzahl: 34999969)" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table4(exampleKeyLength,exampleHashTableSize,universal0);
    example_hash_table4.createCells(1,(int)exampleKeyLength*2);
    example_hash_table4.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "5. Hashfunktion: Universelle Hashfunktion" << std::endl;
    std::cout << "                 (a: 15999950  b: 15999990  Primzahl: 15999989)" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table5(exampleKeyLength,exampleHashTableSize,universal1);
    example_hash_table5.createCells(1,(int)exampleKeyLength*2);
    example_hash_table5.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "6. Hashfunktion: Universelle Hashfunktion" << std::endl;
    std::cout << "                 (a: 135  b: 140  Primzahl: 149)" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table6(exampleKeyLength,exampleHashTableSize,universal2);
    example_hash_table6.createCells(1,(int)exampleKeyLength*2);
    example_hash_table6.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //1. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "7. Hashfunktion: DyCuckoo-Hash 1" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table7(exampleKeyLength,exampleHashTableSize,dycuckoo_hash1);
    example_hash_table7.createCells(1,(int)exampleKeyLength*2);
    example_hash_table7.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "9. Hashfunktion: DyCuckoo-Hash 2" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table8(exampleKeyLength,exampleHashTableSize,dycuckoo_hash2);
    example_hash_table8.createCells(1,(int)exampleKeyLength*2);
    example_hash_table8.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "9. Hashfunktion: DyCuckoo-Hash 3" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table9(exampleKeyLength,exampleHashTableSize,dycuckoo_hash3);
    example_hash_table9.createCells(1,(int)exampleKeyLength*2);
    example_hash_table9.insertTestCells(no_probe);


    /////////////////////////////////////////////////////////////////////////////////////////
    //4. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "10. Hashfunktion: DyCuckoo-Hash 4" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table10(exampleKeyLength,exampleHashTableSize,dycuckoo_hash4);
    example_hash_table10.createCells(1,(int)exampleKeyLength*2);
    example_hash_table10.insertTestCells(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //5. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "11. Hashfunktion: DyCuckoo-Hash 5" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table11(exampleKeyLength,exampleHashTableSize,dycuckoo_hash5);
    example_hash_table11.createCells(1,(int)exampleKeyLength*2);
    example_hash_table11.insertTestCells(no_probe);

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer für alle Hashfunktionen ohne    : ";
    std::cout << timer.getDuration() << std::endl;
    std::cout << "Kollionsauflösung (in Millisekunden)" << std::endl;
    
    return 0;
};