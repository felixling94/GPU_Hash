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
//Leistungsvergleich zwischen verschiedenen Hashfunktionen
/////////////////////////////////////////////////////////////////////////////////////////
int main(){
    //1. Deklariere die Variablen
    const size_t block_num{128}, num_threads_per_block{128};

    const size_t exampleKeyNum{block_num*num_threads_per_block};
    const size_t exampleHashTableSize{exampleKeyNum};

    const size_t matrix_size{exampleKeyNum * sizeof(cell<uint32_t,uint32_t>)};

    int deviceID{0};
    struct cudaDeviceProp props;

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "GPU" << "," << props.name << std::endl;
    std::cout << "VRAM" << "," << (props.totalGlobalMem/1024)/1024 << "MB" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten" << ",";
    std::cout << ((matrix_size * 3 + sizeof(cell<uint32_t,uint32_t>)) / 1024 / 1024) << "MB\n" << std::endl;
    std::cout << "Block_Zahl" << "," << "Threads_Zahl_Pro_Block" << std::endl;
    std::cout << block_num << "," << num_threads_per_block << std::endl;
    std::cout << std::endl;    
    std::cout << "Anzahl der gespeicherten Zellen" << "," << exampleKeyNum << std::endl;
    std::cout << "Größe der Hashtabelle" << "," << exampleHashTableSize << std::endl;

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Modulo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "1. Hashfunktion" << "," << "Divisions-Rest-Methode" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table1(exampleKeyNum,0,exampleHashTableSize,modulo,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table1.createCells();
    example_hash_table1.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //Multiplikative Methode
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "2. Hashfunktion" << "," << "Multiplikative Methode" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table2(exampleKeyNum,0,exampleHashTableSize,multiplication,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table2.createCells();
    example_hash_table2.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //Murmer-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "3. Hashfunktion" << "," << "Murmer-Hashfunktion" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table3(exampleKeyNum,0,exampleHashTableSize,murmer,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table3.createCells();
    example_hash_table3.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //1. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "4. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
    std::cout << "," << "a: 20019" << "," << "b: 20025" << "," <<  "Primzahl: 20029" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table4(exampleKeyNum,0,exampleHashTableSize,universal0,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table4.createCells();
    example_hash_table4.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "5. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
    std::cout << "," << "a: 10023" << "," << "b: 10037" << "," <<  "Primzahl: 10039" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table5(exampleKeyNum,0,exampleHashTableSize,universal1,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table5.createCells();
    example_hash_table5.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "6. Hashfunktion" << "," << "Universelle Hashfunktion" << std::endl;
    std::cout << "," << "a: 5029" << "," << "b: 5038" << "," <<  "Primzahl: 5039" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table6(exampleKeyNum,0,exampleHashTableSize,universal2,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table6.createCells();
    example_hash_table6.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //1. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "7. Hashfunktion" << "," << "DyCuckoo-1" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table7(exampleKeyNum,0,exampleHashTableSize,dycuckoo_hash1,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table7.createCells();
    example_hash_table7.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "9. Hashfunktion" << "," << "DyCuckoo-2" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table8(exampleKeyNum,0,exampleHashTableSize,dycuckoo_hash2,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table8.createCells();
    example_hash_table8.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "9. Hashfunktion" << "," << "DyCuckoo-3" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table9(exampleKeyNum,0,exampleHashTableSize,dycuckoo_hash3,modulo,
                                                              num_threads_per_block,block_num);
    example_hash_table9.createCells();
    example_hash_table9.insertTestCells2(no_probe);


    /////////////////////////////////////////////////////////////////////////////////////////
    //4. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "10. Hashfunktion" << "," << "DyCuckoo-4" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table10(exampleKeyNum,0,exampleHashTableSize,dycuckoo_hash4,modulo,
                                                               num_threads_per_block,block_num);
    example_hash_table10.createCells();
    example_hash_table10.insertTestCells2(no_probe);

    /////////////////////////////////////////////////////////////////////////////////////////
    //5. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::endl;
    std::cout << "11. Hashfunktion" << "," << "DyCuckoo-5" << std::endl;
    std::cout << std::endl;

    Example_Hash_Table<uint32_t,uint32_t> example_hash_table11(exampleKeyNum,0,exampleHashTableSize,dycuckoo_hash5,modulo,
                                                               num_threads_per_block,block_num);
    example_hash_table11.createCells();
    example_hash_table11.insertTestCells2(no_probe);

    //Fasse Resultate zusammen
    timer.stop();

    std::cout << std::endl;
    std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;
    std::cout << std::endl;
   
    return 0;
};