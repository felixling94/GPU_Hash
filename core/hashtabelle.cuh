#ifndef HASHTABELLE_CUH
#define HASHTABELLE_CUH

#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../data/datenvorlage.h>
#include <../include/hashtabelle_device.cuh>
#include <../include/deklaration.cuh>
#include <../core/hashverfahren.cuh>

template <typename T1, typename T2>
GLOBALQUALIFIER void insert0_kernel(Zelle<T1,T2> * pZellen, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle){
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    //int i = threadIdx.x;
    
    insert0<T1,T2>(pZellen[i].schluessel, pZellen[i].wert, pHashtabelle, pGroesseHashtabelle);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert1_kernel(Zelle<T1,T2> * pZellen, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle){
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    //int i = threadIdx.x;

    insert1<T1,T2>(pZellen[i].schluessel, pZellen[i].wert, pHashtabelle, pGroesseHashtabelle);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert2_kernel(Zelle<T1,T2> * pZellen, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle){
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    //int i = threadIdx.x;
     
    insert2<T1,T2>(pZellen[i].schluessel, pZellen[i].wert, pHashtabelle, pGroesseHashtabelle);
};

template <typename T1, typename T2>
GLOBALQUALIFIER void insert3_kernel(Zelle<T1,T2> * pZellen, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle){
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    //int i = threadIdx.x;
    
    insert3<T1,T2>(pZellen[i].schluessel, pZellen[i].wert, pHashtabelle, pGroesseHashtabelle);
};

template <typename T1, typename T2>
Hashtabelle_Device<T1,T2>::Hashtabelle_Device():
hashtyp_kode(keine_aufloesung),groesseHashtabelle(2){
    hashtabelle = new Zelle<T1,T2>[2];
};

template <typename T1, typename T2>
Hashtabelle_Device<T1,T2>::Hashtabelle_Device(hashtyp pHashtyp, size_t pGroesse):
hashtyp_kode(pHashtyp),groesseHashtabelle(pGroesse){
    hashtabelle = new Zelle<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Hashtabelle_Device<T1,T2>::~Hashtabelle_Device(){
    delete[] hashtabelle;
};

//Drucke die Zeile einer Hashtabelle
template <typename T1, typename T2>
std::string Hashtabelle_Device<T1,T2>::getZelle(size_t pIndex){
    std::string zeichenkette;

    if (pIndex < (groesseHashtabelle)){
        if (hashtabelle[pIndex].schluessel!= 0){
            zeichenkette.append(std::to_string(hashtabelle[pIndex].schluessel));
            zeichenkette.append("  ");
            zeichenkette.append(std::to_string(hashtabelle[pIndex].wert));
        }else{
            zeichenkette.append("Leer      Leer");
        } 
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten ohne Kollisionsauflösung gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2>::insert0(Zelle<T1,T2> * zellen, size_t pGroesse){
    Zelle<T1,T2> * zellen_device;
    Zelle<T1,T2> * hashtabelle_device;

    //Erstelle Ereignisse, um Dauer für GPU zu messen
    cudaStream_t stream1, stream2;
    cudaStream_t stream3;
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEvent_t start3, stop3;
    
    cudaStreamCreate(&stream1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    cudaStreamCreate(&stream2);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaStreamCreate(&stream3);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
    cudaMalloc(&hashtabelle_device,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
    cudaMalloc(&zellen_device,sizeof(Zelle<T1,T2>)*pGroesse);

    cudaEventRecord(start1,stream1);
    cudaMemcpyAsync(hashtabelle_device,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(zellen_device,zellen,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice,stream1);
    cudaEventRecord(stop1,stream1);
    cudaEventSynchronize(stop1); 
    
    //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
    int min_gridgroesse, gridgroesse, block_groesse;
    cudaOccupancyMaxPotentialBlockSize(&min_gridgroesse, &block_groesse, insert0_kernel<T1,T2>, 0, 0);
    gridgroesse = ((size_t)(pGroesse)+block_groesse-1)/block_groesse;
    dim3 block(block_groesse);
    dim3 grid(gridgroesse);
    
    void *args[3] = {&zellen_device, &hashtabelle_device, &groesseHashtabelle};

    cudaEventRecord(start2,stream2);
    cudaLaunchKernel((void*)insert0_kernel<T1,T2>,grid,block,args,0,stream2);
    cudaEventRecord(stop2,stream2); 
    cudaEventSynchronize(stop2);

    //Kopiere Daten aus der GPU zur Hashtabelle
    cudaEventRecord(start3,stream3);
    cudaMemcpyAsync(hashtabelle, hashtabelle_device, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost,stream3);
    cudaEventRecord(stop3,stream3);
    cudaEventSynchronize(stop3); 
    
    float dauer_hochladen = 0;
    float dauer_ausfuehrung = 0;
    float dauer_herunterladen = 0;
    float dauer_gesamt = 0;
    
    cudaEventElapsedTime(&dauer_hochladen, start1, stop1);
    cudaEventElapsedTime(&dauer_ausfuehrung, start2, stop2);
    cudaEventElapsedTime(&dauer_herunterladen, start3, stop3);
    cudaEventElapsedTime(&dauer_gesamt, start1, stop3);
    
    std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
    std::cout <<  dauer_hochladen << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)     : ";
    std::cout <<  dauer_ausfuehrung << std::endl;
    std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
    std::cout <<  dauer_herunterladen << std::endl;
    std::cout << "Gesamtdauer (in Millisekunden)              : ";
    std::cout <<  dauer_gesamt << std::endl;
    
    cudaFree(hashtabelle_device);
    cudaFree(zellen_device);
    cudaFree(zellen);
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten mit linearen Hashverfahren gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2>::insert_linear(Zelle<T1,T2> * zellen, size_t pGroesse){
    Zelle<T1,T2> * zellen_device;
    Zelle<T1,T2> * hashtabelle_device;

    //Erstelle Ereignisse, um Dauer für GPU zu messen
    cudaStream_t stream1, stream2;
    cudaStream_t stream3;
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEvent_t start3, stop3;
    
    cudaStreamCreate(&stream1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    cudaStreamCreate(&stream2);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaStreamCreate(&stream3);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
    cudaMalloc(&hashtabelle_device,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
    cudaMalloc(&zellen_device,sizeof(Zelle<T1,T2>)*pGroesse);

    cudaEventRecord(start1,stream1);
    cudaMemcpyAsync(hashtabelle_device,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(zellen_device,zellen,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice,stream1);
    cudaEventRecord(stop1,stream1);
    cudaEventSynchronize(stop1); 

    //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
    int min_gridgroesse, gridgroesse, block_groesse;
    cudaOccupancyMaxPotentialBlockSize(&min_gridgroesse, &block_groesse, insert0_kernel<T1,T2>, 0, 0);
    gridgroesse = ((size_t)(pGroesse)+block_groesse-1)/block_groesse;
    dim3 block(block_groesse);
    dim3 grid(gridgroesse);
    
    void *args[3] = {&zellen_device, &hashtabelle_device, &groesseHashtabelle};

    cudaEventRecord(start2,stream2);
    cudaLaunchKernel((void*)insert1_kernel<T1,T2>,grid,block,args,0,stream2);
    cudaEventRecord(stop2,stream2); 
    cudaEventSynchronize(stop2);

    //Kopiere Daten aus der GPU zur Hashtabelle
    cudaEventRecord(start3,stream3);
    cudaMemcpyAsync(hashtabelle, hashtabelle_device, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost,stream3);
    cudaEventRecord(stop3,stream3);
    cudaEventSynchronize(stop3); 
    
    float dauer_hochladen = 0;
    float dauer_ausfuehrung = 0;
    float dauer_herunterladen = 0;
    float dauer_gesamt = 0;
    
    cudaEventElapsedTime(&dauer_hochladen, start1, stop1);
    cudaEventElapsedTime(&dauer_ausfuehrung, start2, stop2);
    cudaEventElapsedTime(&dauer_herunterladen, start3, stop3);
    cudaEventElapsedTime(&dauer_gesamt, start1, stop3);
     
    std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
    std::cout <<  dauer_hochladen << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)     : ";
    std::cout <<  dauer_ausfuehrung << std::endl;
    std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
    std::cout <<  dauer_herunterladen << std::endl;
    std::cout << "Gesamtdauer (in Millisekunden)              : ";
    std::cout <<  dauer_gesamt << std::endl;
    
    cudaFree(hashtabelle_device);
    cudaFree(zellen_device);
    cudaFree(zellen);
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten mit quadratischen Hashverfahren gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2>::insert_quadratisch(Zelle<T1,T2> * zellen, size_t pGroesse){
    Zelle<T1,T2> * zellen_device;
    Zelle<T1,T2> * hashtabelle_device;

    //Erstelle Ereignisse, um Dauer für GPU zu messen
    cudaStream_t stream1, stream2;
    cudaStream_t stream3;
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEvent_t start3, stop3;
    
    cudaStreamCreate(&stream1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    cudaStreamCreate(&stream2);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaStreamCreate(&stream3);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
    cudaMalloc(&hashtabelle_device,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
    cudaMalloc(&zellen_device,sizeof(Zelle<T1,T2>)*pGroesse);

    cudaEventRecord(start1,stream1);
    cudaMemcpyAsync(hashtabelle_device,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(zellen_device,zellen,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice,stream1);
    cudaEventRecord(stop1,stream1);
    cudaEventSynchronize(stop1); 
    
    //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
    int min_gridgroesse, gridgroesse, block_groesse;
    cudaOccupancyMaxPotentialBlockSize(&min_gridgroesse, &block_groesse, insert0_kernel<T1,T2>, 0, 0);
    gridgroesse = ((size_t)(pGroesse)+block_groesse-1)/block_groesse;
    dim3 block(block_groesse);
    dim3 grid(gridgroesse);
    
    void *args[3] = {&zellen_device, &hashtabelle_device, &groesseHashtabelle};

    cudaEventRecord(start2,stream2);
    cudaLaunchKernel((void*)insert2_kernel<T1,T2>,grid,block,args,0,stream2);
    cudaEventRecord(stop2,stream2); 
    cudaEventSynchronize(stop2);

    //Kopiere Daten aus der GPU zur Hashtabelle
    cudaEventRecord(start3,stream3);
    cudaMemcpyAsync(hashtabelle, hashtabelle_device, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost,stream3);
    cudaEventRecord(stop3,stream3);
    cudaEventSynchronize(stop3); 
    
    float dauer_hochladen = 0;
    float dauer_ausfuehrung = 0;
    float dauer_herunterladen = 0;
    float dauer_gesamt = 0;
    
    cudaEventElapsedTime(&dauer_hochladen, start1, stop1);
    cudaEventElapsedTime(&dauer_ausfuehrung, start2, stop2);
    cudaEventElapsedTime(&dauer_herunterladen, start3, stop3);
    cudaEventElapsedTime(&dauer_gesamt, start1, stop3);
    
    std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
    std::cout <<  dauer_hochladen << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)     : ";
    std::cout <<  dauer_ausfuehrung << std::endl;
    std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
    std::cout <<  dauer_herunterladen << std::endl;
    std::cout << "Gesamtdauer (in Millisekunden)              : ";
    std::cout <<  dauer_gesamt << std::endl;
    
    cudaFree(hashtabelle_device);
    cudaFree(zellen_device);
    cudaFree(zellen);
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten mit doppelten Hashverfahren gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2,>::insert_doppelt(Zelle<T1,T2> * zellen, size_t pGroesse){
    Zelle<T1,T2> * zellen_device;
    Zelle<T1,T2> * hashtabelle_device;

    //Erstelle Ereignisse, um Dauer für GPU zu messen
    cudaStream_t stream1, stream2;
    cudaStream_t stream3;
    cudaEvent_t start1, stop1, start2, stop2; 
    cudaEvent_t start3, stop3;
    
    cudaStreamCreate(&stream1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    cudaStreamCreate(&stream2);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaStreamCreate(&stream3);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
    cudaMalloc(&hashtabelle_device,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
    cudaMalloc(&zellen_device,sizeof(Zelle<T1,T2>)*pGroesse);

    cudaEventRecord(start1,stream1);
    cudaMemcpyAsync(hashtabelle_device,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(zellen_device,zellen,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice,stream1);
    cudaEventRecord(stop1,stream1);
    cudaEventSynchronize(stop1); 
    
    //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
    int min_gridgroesse, gridgroesse, block_groesse;
    cudaOccupancyMaxPotentialBlockSize(&min_gridgroesse, &block_groesse, insert0_kernel<T1,T2>, 0, 0);
    gridgroesse = ((size_t)(pGroesse)+block_groesse-1)/block_groesse;
    dim3 block(block_groesse);
    dim3 grid(gridgroesse);
    
    void *args[3] = {&zellen_device, &hashtabelle_device, &groesseHashtabelle};

    cudaEventRecord(start2,stream2);
    cudaLaunchKernel((void*)insert3_kernel<T1,T2>,grid,block,args,0,stream2);
    cudaEventRecord(stop2,stream2); 
    cudaEventSynchronize(stop2);  
    
    //Kopiere Daten aus der GPU zur Hashtabelle
    cudaEventRecord(start3,stream3);
    cudaMemcpyAsync(hashtabelle, hashtabelle_device, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost,stream3);
    cudaEventRecord(stop3,stream3);
    cudaEventSynchronize(stop3); 
    
    float dauer_hochladen = 0;
    float dauer_ausfuehrung = 0;
    float dauer_herunterladen = 0;
    float dauer_gesamt = 0;
    
    cudaEventElapsedTime(&dauer_hochladen, start1, stop1);
    cudaEventElapsedTime(&dauer_ausfuehrung, start2, stop2);
    cudaEventElapsedTime(&dauer_herunterladen, start3, stop3);
    cudaEventElapsedTime(&dauer_gesamt, start1, stop3);
    
    std::cout << "Dauer zum Hochladen (in Millisekunden)      : ";
    std::cout <<  dauer_hochladen << std::endl;
    std::cout << "Dauer zur Ausführung (in Millisekunden)     : ";
    std::cout <<  dauer_ausfuehrung << std::endl;
    std::cout << "Dauer zum Herunterladen (in Millisekunden)  : ";
    std::cout <<  dauer_herunterladen << std::endl;
    std::cout << "Gesamtdauer (in Millisekunden)              : ";
    std::cout <<  dauer_gesamt << std::endl;
    
    cudaFree(hashtabelle_device);
    cudaFree(zellen_device);
    cudaFree(zellen);
};

//Gebe die Anzahl der Zellen in der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hashtabelle_Device<T1,T2>::getzahlZellen(){
    size_t zahl = 0;
    for (size_t i=0; i<groesseHashtabelle; i++) if(hashtabelle[i].schluessel!=LeerFeld) ++zahl;
    return zahl;
};

//Gebe die Größe der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hashtabelle_Device<T1,T2>::getGroesseHashtabelle(){
    return groesseHashtabelle;
};

//Gebe die Hashtabelle zurück
template <typename T1, typename T2>
Zelle<T1,T2> * Hashtabelle_Device<T1,T2>::getHashtabelle(){
    return hashtabelle;
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T1, typename T2>
hashtyp Hashtabelle_Device<T1,T2>::getHashTyp(){
    return hashtyp_kode;
};

//Drucke die Hashtabelle
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2>::drucken(){
    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
    for(size_t i = 0; i < (groesseHashtabelle); i++) std::cout << i << "  " << getZelle(i) << std::endl;  
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle_Device<T1,T2>::insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse){
    if(pGroesse > groesseHashtabelle){
        std::cout << "Die Größe der der Hashtabelle hinzufügenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
    }else{
        Zelle<T1,T2> * zellen_neu;
        std::vector<Zelle<T1,T2>> zellen_vektor;

        zellen_vektor.reserve(pGroesse);
        
        for (size_t i = 0; i < pGroesse ; i++)
            zellen_vektor.push_back(Zelle<T1,T2>{pSchluesselListe[i],pWerteListe[i]});

        zellen_neu = zellen_vektor.data();

        if (hashtyp_kode == keine_aufloesung){
            insert0(zellen_neu,pGroesse);
        }else if (hashtyp_kode == linear_aufloesung){
            insert_linear(zellen_neu,pGroesse);
        }else if (hashtyp_kode == quadratisch_aufloesung){
            insert_quadratisch(zellen_neu,pGroesse);
        }else{
            insert_doppelt(zellen_neu,pGroesse);
        }
    }
};

#endif