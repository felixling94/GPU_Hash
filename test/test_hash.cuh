#ifndef TEST_HASH_CUH
#define TEST_HASH_CUH

#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <iterator>
#include <stdint.h>

#include <../include/base.h>
#include <../include/declaration.cuh>
#include <../include/hash_function.cuh>
#include <../tools/timer.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <typename T>
GLOBALQUALIFIER void calculate_hash_kernel(T* A, T* result, size_t tableSize, hash_function hashFunction){
    int i_inBlock, blockID, i;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    result[i] = getHash<T>(A[i],tableSize,hashFunction);
};

template <typename T>
GLOBALQUALIFIER void calculate_perfect_hash_kernel(T* A, T* result, size_t tableSize, size_t a, size_t b, size_t primeNum){
    int i_inBlock, blockID, i;
    
    i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    blockID = blockIdx.x;
    i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);

    result[i] = perfect_hash<T>(A[i],tableSize,a,b,primeNum);
};

template<typename T> 
class Test_Hash{
    private:
        T* testArray;
        T* resultArray;
        T* resultArray_device;

        size_t array_size = 11;
        size_t table_size = 13;

        size_t a;
        size_t b;
        size_t prime_num;

    public:
        Test_Hash(){
            testArray = new T[11];
            resultArray = new T[11];
            resultArray_device = new T[11];
        };

        Test_Hash(size_t keySize, size_t tableSize, size_t p_a=15999950, size_t p_b=15999990,size_t primNum= 15999989):
        array_size(keySize),table_size(tableSize),a(p_a),b(p_b),prime_num(primNum){
            testArray = new T[keySize];
            resultArray = new T[keySize];
            resultArray_device = new T[keySize];
        };

        ~Test_Hash(){
            delete[] testArray;
            delete[] resultArray;
            delete[] resultArray_device;
        };
        
        //Erzeuge verschiedene Schlüssel zufällig
        void createKeys(size_t min=0, size_t max=100){
            //1. Deklariere und initialisiere Variablen
            static std::random_device rd;
            static std::mt19937 mte(rd());
            std::uniform_int_distribution<T> dist(min, max);
    
            //2. Erzeuge zufällige Werte mithilfe Zufallsgenerator
            for (size_t i = 0; i<array_size; ++i) testArray[i] = static_cast<T>(dist(mte));
        };

        //Vergleiche zwischen zwei Arrayelementen
        void compare(T *lhs, T *rhs) {
            //1. Deklariere die Variablen
            int errors{0};
            float lhs_type, rhs_type;
            std::string errors_string, array_size_string, i_string, lhs_string,rhs_string;

            //2A. Berechne die Anzahl von Fehlern, die durch Inkonsistenzen verursacht werden
            for (int i{0}; i < array_size; i += 1) {
                lhs_type = static_cast<int>(lhs[i]);
                rhs_type = static_cast<int>(rhs[i]);
                
                if ((lhs_type - rhs_type) != 0) {
                    errors += 1;
                    std::cout << i << " erwartet " << lhs[i] << ": tatsächlich " << rhs[i]  << std::endl;
                }
            }
            
            //2B. Bestimme, ob es Fehler bei den Wertkonsistenzen von zwei Arrayelementen gibt
            errors_string = static_cast<int>(errors);
            array_size_string = static_cast<int>(array_size);
            
            if (errors > 0) {
                std::cout << errors_string << " Fehlern verursacht, aus " << array_size_string << " Werten." <<  std::endl;
            } else {
                std::cout << "Keine Fehler gefunden." << std::endl;
            }
        };
        
        //Vergleiche zwischen zwei Arrayelementen
        std::function<bool(const T &, const T &)> comparator =[](const T &left, const T &right){
            double epsilon{1.0E-8};
            float lhs_type = static_cast<float>(left), rhs_type = static_cast<float>(right);
            return (abs(lhs_type - rhs_type) < epsilon);
        };

        //Gebe die Größe der Testfelder zurück
        size_t getArraySize(){
            return array_size;
        };

        //Gebe die Größe einer Hashtabelle zurück
        size_t getTableSize(){
            return table_size;
        };
        
        //Gebe die Felder zurück
        T* getArray(){
            return testArray;
        };
        
        //Gebe den Hashwert eines Schlüssels zurück
        T getHashValue(T key, hash_function function){
            if (function == perfect3){
                return perfect_hash<T>(key,table_size,a,b,prime_num);
            }else{
                return getHash<T>(key,table_size,function);
            }
        }
        
        //Berechne die Hashwerte auf dem Host
        void getHashArrayOnHost(hash_function function){
            //1. Deklariere und initialisiere die Variablen   
            CPUTimer timer;

            timer.start();

            //2. Führe Schleifen aus, um Hashwerte zu bestimmen
            for (size_t i = 0; i<array_size; ++i) resultArray[i] = static_cast<T>(getHashValue(testArray[i],function));

            timer.stop();
            std::cout << "Dauer zur Ausführung (in ms)                : ";
            std::cout <<  timer.getDuration() << std::endl;
        };

        void getHashArrayOnDevice(hash_function function){
            float duration_upload, duration_run, duration_download, duration_total;
            int min_grid_size, grid_size, block_size;
            GPUTimer upload, run, download;
        
            duration_upload = 0; 
            duration_run = 0; 
            duration_download = 0;
            duration_total = 0;
            
            T * testArray_Device;
            T * testResultArray_Device;
            
            //Reserviere und kopiere Daten aus der TestArray und eingegebenen Zellen auf GPU
            cudaMalloc(&testArray_Device,(sizeof(T))*array_size);
            cudaMalloc(&testResultArray_Device,(sizeof(T))*array_size);

            upload.GPUstart();
            cudaMemcpyAsync(testArray_Device,testArray,(sizeof(T))*array_size,cudaMemcpyHostToDevice,upload.getStream());      
            cudaMemcpyAsync(testResultArray_Device,resultArray_device,(sizeof(T))*array_size,cudaMemcpyHostToDevice,upload.getStream());
            upload.GPUstop();

            //Berechnen die Hashwerte durch Device
            if (function == perfect0 || function == perfect1 ||function == perfect2 || function == perfect3){
                cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_perfect_hash_kernel<T>, 0, 0);
                grid_size = ((size_t)(array_size)+block_size-1)/block_size;
                dim3 block(block_size);
                dim3 grid(grid_size);
                
                void *args[6] = {&testArray_Device, &testResultArray_Device, &table_size,&a,&b,&prime_num};
                run.GPUstart();
                cudaLaunchKernel((void*)calculate_perfect_hash_kernel<T>,grid,block,args,0,run.getStream());
                run.GPUstop(); 

            }else{
                cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_hash_kernel<T>, 0, 0);
                grid_size = ((size_t)(array_size)+block_size-1)/block_size;
                dim3 block(block_size);
                dim3 grid(grid_size);
                
                void *args[4] = {&testArray_Device, &testResultArray_Device, &table_size,&function};
                run.GPUstart();
                cudaLaunchKernel((void*)calculate_hash_kernel<T>,grid,block,args,0,run.getStream());
                run.GPUstop(); 
            }

            //Kopiere Daten aus der GPU zur TestArray
            download.GPUstart();
            cudaMemcpyAsync(resultArray_device, testResultArray_Device, sizeof(T)*array_size, cudaMemcpyDeviceToHost,download.getStream());
            download.GPUstop();
        
            duration_upload = upload.getGPUDuration();
            duration_run = run.getGPUDuration();
            duration_download = download.getGPUDuration();
            duration_total = duration_upload + duration_run + duration_download;

            std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
            std::cout <<  duration_upload << std::endl;
            std::cout << "Dauer zur Ausführung (in Millisekunden)     : ";
            std::cout <<  duration_run << std::endl;
            std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
            std::cout <<  duration_download << std::endl;
            std::cout << "Gesamtdauer (in Millisekunden)              : ";
            std::cout <<  duration_total << std::endl;
    
            cudaFree(testArray_Device);
            cudaFree(testResultArray_Device);
        };

        void compare_host_device(hash_function function){
            //1. Deklariere und initialisiere alle Variablen
            bool equalHashArray;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (function == multiplication){
                std::cout << "Multiplikative Methode" << std::endl;
            }else if (function == perfect0 || function == perfect1 || function == perfect2 || function == perfect3){
                std::cout << "Perfekte Hashverfahren (a: " << a << "  b: " << b << "  Primzahl: " << prime_num << ")" << std::endl;
            }else if (function == murmer){
                std::cout << "Murmer Hash" << std::endl;
            }else if (function == dycuckoo_hash1){
                std::cout << "DyCuckoo Hash 1" << std::endl;
            }else if (function == dycuckoo_hash2){
                std::cout << "DyCuckoo Hash 2" << std::endl;
            }else if (function ==dycuckoo_hash3){
                std::cout << "DyCuckoo Hash 3" << std::endl;
            }else if (function ==dycuckoo_hash4){
                std::cout << "DyCuckoo Hash 4" << std::endl;
            }else if (function ==dycuckoo_hash5){
                std::cout << "DyCuckoo Hash 5" << std::endl;
            }else{
                std::cout << "Divisionsmethode" << std::endl;
            }
            
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
        
            getHashArrayOnHost(function);
            
            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << std::endl;
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            getHashArrayOnDevice(function);

            //Vergleiche zwischen zwei Arrays
            equalHashArray = std::equal(resultArray_device, resultArray_device + (array_size), resultArray, comparator);
            if (!equalHashArray) compare(resultArray, resultArray_device);
            
            cudaFree(resultArray_device);
        };
};

#endif
