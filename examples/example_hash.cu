#include <iostream>
#include <stdint.h>

#include <../include/base.h>
#include "example_hash.cuh"

int main(){
    //1. Deklariere und initialisiere Variablen
    const size_t block_num{1024}, num_threads_per_block{1024};
    const size_t example_array_size{block_num*num_threads_per_block};
    size_t example_table_size{(size_t) (example_array_size*120/100)};

    const size_t matrix_size{example_array_size*sizeof(uint32_t)};
    
    int deviceID{0};
    struct cudaDeviceProp props;

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "GPU" << "," << props.name << std::endl;
    std::cout << "VRAM" << "," << (props.totalGlobalMem/1024)/1024 << "MB" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten" << ",";
    std::cout << ((matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "MB\n" << std::endl;
    std::cout << "Block_Zahl" << "," << "Threads_Zahl_Pro_Block" << std::endl;
    std::cout << block_num << "," << num_threads_per_block << std::endl;
    std::cout << std::endl;   
    std::cout << "Anzahl der Schlüssel" << "," << example_array_size << std::endl;
    std::cout << "Größe der Hashtabelle" << "," << example_table_size << std::endl;
    std::cout << std::endl; 

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Modulo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash1(example_array_size,example_table_size,0,0,0,
                                         num_threads_per_block, block_num);
    example_hash1.createKeys();
    example_hash1.compare_host_device(modulo);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Multiplikative Methode
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash2(example_array_size,example_table_size,0,0,0,
                                         num_threads_per_block, block_num);
    example_hash2.createKeys();
    example_hash2.compare_host_device(multiplication);
    /////////////////////////////////////////////////////////////////////////////////////////
    //1. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash3(example_array_size,example_table_size,20019,20025,20029,
                                         num_threads_per_block, block_num);
    example_hash3.createKeys();
    example_hash3.compare_host_device(universal0);
    /////////////////////////////////////////////////////////////////////////////////////////
    //2. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash4(example_array_size,example_table_size,10023,10037,10039,
                                         num_threads_per_block, block_num);
    example_hash4.createKeys();
    example_hash4.compare_host_device(universal1);
    /////////////////////////////////////////////////////////////////////////////////////////
    //3. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash5(example_array_size,example_table_size,5029,5038,5039,
                                         num_threads_per_block, block_num);
    example_hash5.createKeys();
    example_hash5.compare_host_device(universal2);
    /////////////////////////////////////////////////////////////////////////////////////////
    //4. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash6(example_array_size,example_table_size,50,60,71,
                                         num_threads_per_block, block_num);
    example_hash6.createKeys();
    example_hash6.compare_host_device(universal3);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Murmer-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash7(example_array_size,example_table_size,0,0,0,
                                         num_threads_per_block, block_num);
    example_hash7.createKeys();
    example_hash7.compare_host_device(murmer);


    /////////////////////////////////////////////////////////////////////////////////////////
    //1. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash8(example_array_size,example_table_size,0,0,0,
                                         num_threads_per_block, block_num);
    example_hash8.createKeys();
    example_hash8.compare_host_device(dycuckoo_hash1);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash9(example_array_size,example_table_size,0,0,0,
                                         num_threads_per_block, block_num);
    example_hash9.createKeys();
    example_hash9.compare_host_device(dycuckoo_hash2);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash10(example_array_size,example_table_size,0,0,0,
                                          num_threads_per_block, block_num);
    example_hash10.createKeys();
    example_hash10.compare_host_device(dycuckoo_hash3);

    /////////////////////////////////////////////////////////////////////////////////////////
    //4. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash11(example_array_size,example_table_size,0,0,0,
                                          num_threads_per_block, block_num);
    example_hash11.createKeys();
    example_hash11.compare_host_device(dycuckoo_hash4);
    /////////////////////////////////////////////////////////////////////////////////////////
    //5. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t> example_hash12(example_array_size,example_table_size,0,0,0,
                                          num_threads_per_block, block_num);
    example_hash12.createKeys();
    example_hash12.compare_host_device(dycuckoo_hash5);

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer" << "," << timer.getDuration() << std::endl;

    return 0;
};