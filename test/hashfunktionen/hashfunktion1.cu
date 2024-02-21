#include <iostream>
#include <string>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>
#include <stdint.h>

#include <../data/datenvorlage.h>
#include <../include/deklaration.cuh>
#include <../hashfunktionen/hashfunktionen.h>
#include <../hashfunktionen/hashfunktionen.cuh>
#include <../tools/timer.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

GLOBALQUALIFIER void hash_berechnen(uint32_t * A, uint32_t * B, size_t hashtabellegroesse, hashfunktion kode){
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    __syncthreads();
    if (kode == murmer){
        B[i] = Hashfunktionen_Device::murmer_hash<uint32_t>(A[i],hashtabellegroesse);
    }else if (kode == multiplikativ){
        B[i] = Hashfunktionen_Device::multiplikativ_hash<uint32_t>(A[i],hashtabellegroesse);
    }else{
        B[i] = Hashfunktionen_Device::modulo_hash<uint32_t>(A[i],hashtabellegroesse);
    }
    __syncthreads();
};

//Vorlage-Funktion zum Vergleich zwischen zwei Arrayelementen
template <typename type> void compare(type *lhs, type *rhs, int array_size) {
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

//Vorlage-Funktion zum Vergleich zwischen zwei Arrayelementen
template <typename type>
std::function<bool(const type &, const type &)> comparator =[](const type &left, const type &right){
    double epsilon{1.0E-8};
    float lhs_type = static_cast<float>(left), rhs_type = static_cast<float>(right);
    return (abs(lhs_type - rhs_type) < epsilon);
};

//Erzeuge verschiedene Schlüssel zufällig
void erzeuge_schluessel(uint32_t *array, size_t array_groesse, size_t min=0, size_t max=100){ 
    //1. Deklariere und initialisiere Variablen
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_int_distribution<uint32_t> dist(min, max);

    //2. Erzeuge zufällige Werte mithilfe Zufallsgenerator
    for (size_t i = 0; i<array_groesse; ++i) array[i] = static_cast<uint32_t>(dist(mte));
};

size_t berechneHashwert(uint32_t schluessel, size_t groesseHashtabelle, hashfunktion kode){
    if (kode == murmer){
        return Hashfunktionen::murmer_hash<uint32_t>(schluessel,groesseHashtabelle);
    }else if (kode == multiplikativ){
        return Hashfunktionen::multiplikativ_hash<uint32_t>(schluessel,groesseHashtabelle);
    }else{
        return Hashfunktionen::modulo_hash<uint32_t>(schluessel,groesseHashtabelle);
    }
};

//Vorlage-Funktion zur Berechnung von Hashwerten
void hashArrayOnHost(uint32_t *A, uint32_t *B, const size_t nx, const size_t ny, const size_t nz, hashfunktion kode) {
    //1. Deklariere die Variablen   
    size_t i;

    //2. Führe Schleifen aus, um Hashwerte zu bestimmen
    for (size_t ix = 0; ix < nx; ++ix) {
        for (size_t iy = 0; iy < ny; ++iy) {
            for (size_t iz = 0; iz < nz; ++iz) {
                i = iz * (nz * ny) + iy * ny + ix;
                B[i] = static_cast<uint32_t>(berechneHashwert(A[i],nx * ny * nz,kode));
            }
        }
    }
    std::cout << std::endl;
};

int main(){
    //1. Deklariere und initialisiere Variablen
    bool equalHashArray1;
    bool equalHashArray2;
    bool equalHashArray3;
    const size_t NX{800}, NY{200}, NZ{200};
    const size_t BLOCK_GROESSE{8};
    const size_t GRID_GROESSE{4};
    const size_t matrix_groesse{(NX * NY * NZ) * sizeof(uint32_t)};
    size_t hashtabelle_groesse{NX*NY*NZ};

    int deviceID{0};
    struct cudaDeviceProp eigenschaften;

    //1A. Überprüfung der Korrektheit von Modulen von NX,NY,NZ mit BLOCK_GROESSE
    static_assert(NX % BLOCK_GROESSE == 0);
    static_assert(NY % BLOCK_GROESSE == 0);
    static_assert(NZ % BLOCK_GROESSE == 0);
    static_assert(NY * GRID_GROESSE == NX);
    static_assert(NZ * GRID_GROESSE == NX);

    uint32_t * A = new uint32_t[NX * NY * NZ];
    uint32_t * B_seq1 = new uint32_t[NX * NY * NZ];
    uint32_t * B_seq2 = new uint32_t[NX * NY * NZ];
    uint32_t * B_seq3 = new uint32_t[NX * NY * NZ];
    uint32_t * B_cuda1 = new uint32_t[NX * NY * NZ];
    uint32_t * B_cuda2 = new uint32_t[NX * NY * NZ];
    uint32_t * B_cuda3 = new uint32_t[NX * NY * NZ];

    //1B. Fülle Arrays mit Zufallswerten aus
    erzeuge_schluessel(A, NX * NY * NZ);
	
    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&eigenschaften, deviceID);

    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Ausgewähltes " << eigenschaften.name << " mit "
              << (eigenschaften.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << ((matrix_groesse * 3 + sizeof(uint32_t)) / 1024 / 1024) << "mb\n" << std::endl;

    //2C. Setze Kernelargumente ein.
    cudaStream_t stream1,stream2,stream3,stream4,stream5;

    cudaStreamCreate (&stream1);
    cudaStreamCreate (&stream2);
    cudaStreamCreate (&stream3);
    cudaStreamCreate (&stream4);
    cudaStreamCreate (&stream5);

    cudaEvent_t start1, start2, start3,start4,start5;
    cudaEvent_t stop1, stop2,stop3,stop4,stop5;
    
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&start4);
    cudaEventCreate(&start5);

    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);
    cudaEventCreate(&stop4);
    cudaEventCreate(&stop5);
    
    uint32_t * A_GPU;
    uint32_t * B_GPU1;
    uint32_t * B_GPU2;
    uint32_t * B_GPU3;

    //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
    cudaMalloc(&A_GPU,matrix_groesse);
    cudaMalloc(&B_GPU1,matrix_groesse);
    cudaMalloc(&B_GPU2,matrix_groesse);
    cudaMalloc(&B_GPU3,matrix_groesse);

    cudaEventRecord(start1,stream1);
    cudaMemcpyAsync(A_GPU,A,matrix_groesse,cudaMemcpyHostToDevice,stream1);      
    cudaMemcpyAsync(B_GPU1,B_cuda1,matrix_groesse,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(B_GPU2,B_cuda2,matrix_groesse,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(B_GPU3,B_cuda3,matrix_groesse,cudaMemcpyHostToDevice,stream1);

    cudaEventRecord(stop1,stream1);
    cudaEventSynchronize(stop1);       
            
    //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
    hashfunktion kode1 = modulo;
    hashfunktion kode2 = murmer;
    hashfunktion kode3 = multiplikativ;
    
    double CPUDauer1 = 0.0;
    double CPUDauer2 = 0.0;
    double CPUDauer3 = 0.0;

    Zeit::starte();
    hashArrayOnHost(A, B_seq1, NX, NY, NZ,kode1);
    Zeit::beende();
    CPUDauer1 = Zeit::getDauer();

    Zeit::starte();
    hashArrayOnHost(A, B_seq2, NX, NY, NZ,kode2);
    Zeit::beende();
    CPUDauer2 = Zeit::getDauer();

    Zeit::starte();
    hashArrayOnHost(A, B_seq3, NX, NY, NZ,kode3);
    Zeit::beende();
    CPUDauer3 = Zeit::getDauer();

    std::cout << "Anzahl von Schlüsseln                       : ";
    std::cout << hashtabelle_groesse << std::endl;
    std::cout << std::endl;

    std::cout << "**********************************************" << std::endl;
    std::cout << "Sequentielle Ausführung" << std::endl;
    std::cout << "**********************************************" << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)" << std::endl;
    std::cout << "Divisionsmethode                            : ";
    std::cout <<  CPUDauer1<< std::endl;
    std::cout << "Murmur-Hashverfahren                        : ";
    std::cout <<  CPUDauer2 << std::endl;
    std::cout << "Multiplikative Methode                      : ";
    std::cout <<  CPUDauer3 << std::endl;
    std::cout <<  std::endl;

    dim3 block(BLOCK_GROESSE, BLOCK_GROESSE, BLOCK_GROESSE);
    dim3 grid(GRID_GROESSE);

    void *args1[4] = {&A_GPU, &B_GPU1, &hashtabelle_groesse,&kode1};
    void *args2[4] = {&A_GPU, &B_GPU2, &hashtabelle_groesse,&kode2};
    void *args3[4] = {&A_GPU, &B_GPU3, &hashtabelle_groesse,&kode3};

    cudaEventRecord(start2,stream2);
    cudaLaunchKernel((void*)hash_berechnen,grid,block,args1,0,stream2);
    cudaEventRecord(stop2,stream2);  
    cudaEventSynchronize(stop2);    

    cudaEventRecord(start3,stream3);
    cudaLaunchKernel((void*)hash_berechnen,grid,block,args2,0,stream3);
    cudaEventRecord(stop3,stream3);  
    cudaEventSynchronize(stop3);   

    cudaEventRecord(start4,stream4);
    cudaLaunchKernel((void*)hash_berechnen,grid,block,args3,0,stream4);
    cudaEventRecord(stop4,stream4);  
    cudaEventSynchronize(stop4);  

    //Kopiere Daten aus der GPU zur Hashtabelle
    cudaEventRecord(start5,stream5);
    cudaMemcpyAsync(B_cuda1, B_GPU1, matrix_groesse, cudaMemcpyDeviceToHost,stream5);
    cudaMemcpyAsync(B_cuda2, B_GPU2, matrix_groesse, cudaMemcpyDeviceToHost,stream5);
    cudaMemcpyAsync(B_cuda3, B_GPU3, matrix_groesse, cudaMemcpyDeviceToHost,stream5);
 
    cudaEventRecord(stop5,stream5);
    cudaEventSynchronize(stop5);

    float streamDauer1 = 0;
    float streamDauer2 = 0;
    float streamDauer3 = 0;
    float streamDauer4 = 0;
    float streamDauer5 = 0;
    float streamDauer6 = 0;

    cudaEventElapsedTime(&streamDauer1, start1, stop1);
    cudaEventElapsedTime(&streamDauer2, start2, stop2);
    cudaEventElapsedTime(&streamDauer3, start3, stop3);
    cudaEventElapsedTime(&streamDauer4, start4, stop4);
    cudaEventElapsedTime(&streamDauer5, start5, stop5);
    cudaEventElapsedTime(&streamDauer6, start1, stop5);

    equalHashArray1 = std::equal(B_cuda1, B_cuda1 + (NX + NY + NY), B_seq1, comparator<uint32_t>);
    equalHashArray2 = std::equal(B_cuda2, B_cuda2 + (NX + NY + NY), B_seq2, comparator<uint32_t>);
    equalHashArray3 = std::equal(B_cuda3, B_cuda3 + (NX + NY + NY), B_seq3, comparator<uint32_t>);

    if (!equalHashArray1) compare<uint32_t>(B_seq1, B_cuda1, NX + NY + NZ);
    if (!equalHashArray2) compare<uint32_t>(B_seq2, B_cuda2, NX + NY + NZ);
    if (!equalHashArray3) compare<uint32_t>(B_seq3, B_cuda3, NX + NY + NZ);

    std::cout << "**********************************************" << std::endl;
    std::cout << "Parallele Ausführung" << std::endl;
    std::cout << "**********************************************" << std::endl;
    std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
    std::cout <<  streamDauer1 << std::endl;
    std::cout << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)" << std::endl;
    std::cout << "Divisionsmethode                            : ";
    std::cout <<  streamDauer2 << std::endl;
    std::cout << "Murmur-Hashverfahren                        : ";
    std::cout <<  streamDauer3 << std::endl;
    std::cout << "Multiplikative Methode                      : ";
    std::cout <<  streamDauer4 << std::endl;
    std::cout << std::endl;
    std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
    std::cout <<  streamDauer5 << std::endl;
    std::cout <<  std::endl;
    std::cout << "Gesamtdauer (in Millisekunden)              : ";
    std::cout <<  streamDauer6 << std::endl;

    //Gebe Ressourcen frei
    delete[] A;
    delete[] B_seq1;
    delete[] B_seq2;
    delete[] B_seq3;
    delete[] B_cuda1;
    delete[] B_cuda2;
    delete[] B_cuda3;
    
    cudaFree(A_GPU);
    cudaFree(B_GPU1);
    cudaFree(B_GPU2);
    cudaFree(B_GPU3);

    return 0;
};