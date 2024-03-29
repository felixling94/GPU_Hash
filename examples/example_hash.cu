#include <iostream>
#include <stdint.h>

#include <../include/base.h>
#include "example_hash.cuh"

int main(){
    //1. Deklariere und initialisiere Variablen
    const size_t example_array_size{800*200*200};
    size_t example_table_size{(size_t) (example_array_size*120/100)};
    const size_t matrix_size{example_array_size*sizeof(uint32_t)};
    
    int deviceID{0};
    struct cudaDeviceProp props;

    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&props, deviceID);

    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Ausgewähltes " << props.name << " mit "
              << (props.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << ((matrix_size * 3 + sizeof(uint32_t)) / 1024 / 1024) << "mb\n" << std::endl;
    
    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;   
    std::cout << "Anzahl der Schlüssel                        : ";
    std::cout << example_array_size << std::endl;
    std::cout << "Größe der Hashtabelle                       : ";
    std::cout << example_table_size << std::endl;

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Modulo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash1(example_array_size,example_table_size);
    example_hash1.createKeys();
    example_hash1.compare_host_device(modulo);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Multiplikative Methode
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash2(example_array_size,example_table_size);
    example_hash2.createKeys();
    example_hash2.compare_host_device(multiplication);
    /////////////////////////////////////////////////////////////////////////////////////////
    //1. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash3(example_array_size,example_table_size,34999950,34999960,34999969);
    example_hash3.createKeys();
    example_hash3.compare_host_device(universal0);
    /////////////////////////////////////////////////////////////////////////////////////////
    //2. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash4(example_array_size,example_table_size,15999950,15999990,15999989);
    example_hash4.createKeys();
    example_hash4.compare_host_device(universal1);
    /////////////////////////////////////////////////////////////////////////////////////////
    //3. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash5(example_array_size,example_table_size,135,140,149);
    example_hash5.createKeys();
    example_hash5.compare_host_device(universal2);
    /////////////////////////////////////////////////////////////////////////////////////////
    //4. Universelle Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash6(example_array_size,example_table_size,50,60,71);
    example_hash6.createKeys();
    example_hash6.compare_host_device(universal3);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Murmer-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash7(example_array_size,example_table_size);
    example_hash7.createKeys();
    example_hash7.compare_host_device(murmer);


    /////////////////////////////////////////////////////////////////////////////////////////
    //1. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash8(example_array_size,example_table_size);
    example_hash8.createKeys();
    example_hash8.compare_host_device(dycuckoo_hash1);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash9(example_array_size,example_table_size);
    example_hash9.createKeys();
    example_hash9.compare_host_device(dycuckoo_hash2);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash10(example_array_size,example_table_size);
    example_hash10.createKeys();
    example_hash10.compare_host_device(dycuckoo_hash3);

    /////////////////////////////////////////////////////////////////////////////////////////
    //4. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash11(example_array_size,example_table_size);
    example_hash11.createKeys();
    example_hash11.compare_host_device(dycuckoo_hash4);
    /////////////////////////////////////////////////////////////////////////////////////////
    //5. DyCuckoo-Hashfunktion
    /////////////////////////////////////////////////////////////////////////////////////////
    Example_Hash<uint32_t>  example_hash12(example_array_size,example_table_size);
    example_hash12.createKeys();
    example_hash12.compare_host_device(dycuckoo_hash5);

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer für alle offenen Hashverfahren  : ";
    std::cout << timer.getDuration() << std::endl;
    std::cout << "(in Millisekunden)" << std::endl;

    return 0;
};