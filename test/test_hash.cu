#include <iostream>
#include <stdint.h>

#include <../include/base.h>
#include "test_hash.cuh"

int main(){
    //1. Deklariere und initialisiere Variablen
    const size_t test_array_size{800*200*200};
    size_t test_table_size{(size_t) (test_array_size*120/100)};
    const size_t matrix_size{test_array_size*sizeof(uint32_t)};
    
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
    std::cout << test_array_size << std::endl;
    std::cout << "Größe der Hashtabelle                       : ";
    std::cout << test_table_size << std::endl;

    CPUTimer timer;
    timer.start();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Modulo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash1(test_array_size,test_table_size);
    test_hash1.createKeys();
    test_hash1.compare_host_device(modulo);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Multiplikative Methode
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash2(test_array_size,test_table_size);
    test_hash2.createKeys();
    test_hash2.compare_host_device(multiplication);
    /////////////////////////////////////////////////////////////////////////////////////////
    //1. Perfekte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash3(test_array_size,test_table_size,34999950,34999960,34999969);
    test_hash3.createKeys();
    test_hash3.compare_host_device(perfect0);
    /////////////////////////////////////////////////////////////////////////////////////////
    //2. Perfekte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash4(test_array_size,test_table_size,15999950,15999990,15999989);
    test_hash4.createKeys();
    test_hash4.compare_host_device(perfect1);
    /////////////////////////////////////////////////////////////////////////////////////////
    //3. Perfekte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash5(test_array_size,test_table_size,135,140,149);
    test_hash5.createKeys();
    test_hash5.compare_host_device(perfect2);
    /////////////////////////////////////////////////////////////////////////////////////////
    //4. Perfekte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash6(test_array_size,test_table_size,50,60,71);
    test_hash6.createKeys();
    test_hash6.compare_host_device(perfect3);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Murmer-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash7(test_array_size,test_table_size);
    test_hash7.createKeys();
    test_hash7.compare_host_device(perfect3);


    /////////////////////////////////////////////////////////////////////////////////////////
    //1. DyCuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash8(test_array_size,test_table_size);
    test_hash8.createKeys();
    test_hash8.compare_host_device(dycuckoo_hash1);

    /////////////////////////////////////////////////////////////////////////////////////////
    //2. DyCuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash9(test_array_size,test_table_size);
    test_hash9.createKeys();
    test_hash9.compare_host_device(dycuckoo_hash2);

    /////////////////////////////////////////////////////////////////////////////////////////
    //3. DyCuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash10(test_array_size,test_table_size);
    test_hash10.createKeys();
    test_hash10.compare_host_device(dycuckoo_hash3);

    /////////////////////////////////////////////////////////////////////////////////////////
    //4. DyCuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash11(test_array_size,test_table_size);
    test_hash11.createKeys();
    test_hash11.compare_host_device(dycuckoo_hash4);
    /////////////////////////////////////////////////////////////////////////////////////////
    //5. DyCuckoo-Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Hash<uint32_t>  test_hash12(test_array_size,test_table_size);
    test_hash12.createKeys();
    test_hash12.compare_host_device(dycuckoo_hash5);

    //Fasse Resultate zusammen
    timer.stop();
    std::cout << std::endl;
    std::cout << "Gesamtdauer für alle offenen Hashverfahren  : ";
    std::cout << timer.getDuration() << std::endl;
    std::cout << "(in Millisekunden)" << std::endl;

    return 0;
};