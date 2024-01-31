#ifndef DYNAMISCH_HASHTABELLE_CUH
#define DYNAMISCH_HASHTABELLE_CUH

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/hashtabelle.h>
#include <../include/dynamisch_hashtabelle.h>
#include <../include/deklaration.cuh>

#include <../src/hashtabelle.cuh>

GLOBALQUALIFIER void kernel_Dynamisch_Insert(Zelle<uint32_t,uint32_t> * pZellen, 
Liste<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle){
    hashtabelle_Dynamisch_Insert<uint32_t,uint32_t>(pZellen,pHashtabelle,pGroesseHashtabelle);
};

GLOBALQUALIFIER void kernel_Dynamisch_Suchen(uint32_t * pSchluesselListe, 
Liste<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle){
    hashtabelle_Dynamisch_Suchen<uint32_t,uint32_t>(pSchluesselListe,pHashtabelle,pGroesseHashtabelle);
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T1, typename T2>
void Dynamisch_Hashtabelle<T1,T2>::insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse){
    Zelle<T1,T2> * zellen_neu;
    Zelle<T1,T2> * zellen_GPU;

    Liste<T1,T2> * hashtabelle_GPU;
    
    if(pGroesse <= this->groesseHashtabelle){
        std::vector<Zelle<T1,T2>> schluesselwertVektor;
        schluesselwertVektor.reserve(pGroesse);
        
        for (size_t i = 0; i < pGroesse ; i++)
            schluesselwertVektor.push_back(Zelle<T1,T2>{pSchluesselListe[i],pWerteListe[i]});
        
        zellen_neu = schluesselwertVektor.data();

    }else{
        std::cout << "Die Größe der der Hashtabelle hinzufügenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
        return;
    }

    if(standAuflosung == true){
        //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
        cudaMalloc(&zellen_GPU,sizeof(Zelle<T1,T2>)*pGroesse);
        cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*sizeof(Liste<T1,T2>)*(this->groesseHashtabelle));
        
        cudaMemcpy(zellen_GPU,zellen_neu,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice);
        cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*sizeof(Liste<T1,T2>)*(this->groesseHashtabelle),cudaMemcpyHostToDevice);
        
        //Erstelle Ereignisse, um Dauer für GPU zu messen
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
        dim3 threads(pGroesse);

        kernel_Dynamisch_Insert<<<1,threads>>>(zellen_GPU, hashtabelle_GPU, (this->groesseHashtabelle));
        
        //Kopiere Daten aus der GPU zur Hashtabelle
        cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*sizeof(Liste<T1,T2>)*(this->groesseHashtabelle), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
  
        float Millisekunden = 0;
        cudaEventElapsedTime(&Millisekunden, start, stop);
        float Sekunden = Millisekunden / 1000.0f;

        std::cout << "GPU hat der Hashtabelle " << pGroesse << " in " << Millisekunden;
        std::cout << " Millisekunden (" << (pGroesse / (double) Sekunden / 1000000.0f);
        std::cout << " Millionen Schlüssel pro Sekunde) bei dynamischen Hashverfahren hinzugefügt." << std::endl;
        
        cudaFree(hashtabelle_GPU);
        cudaFree(zellen_GPU);
        cudaFree(zellen_neu);

        return;

    }else{
        //TODO
        return;
    }
};

//Suche nach einem Array von Schlüsseln in der Hashtabelle gleichzeitig.
template <typename T1, typename T2>
void Dynamisch_Hashtabelle<T1,T2>::suchen_List(T1 * pSchluesselListe, size_t pGroesse){
    T1 * schluesselListe_GPU;
    Liste<T1,T2> * hashtabelle_GPU;
    
    if(pGroesse > (this->groesseHashtabelle)){
        std::cout << "Die Größe der der Hashtabelle nach suchenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
        return;
    }
    
    if (standAuflosung == true){
        //Reserviere und kopiere Daten aus der Hashtabelle, eingegebenen Zellen auf GPU
        cudaMalloc(&schluesselListe_GPU,sizeof(T1)*pGroesse);
        cudaMalloc(&hashtabelle_GPU,sizeof(Liste<T1,T2>)*(this->groesseHashtabelle));
        
        cudaMemcpy(schluesselListe_GPU,pSchluesselListe,sizeof(T1)* pGroesse,cudaMemcpyHostToDevice);
        cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Liste<T1,T2>)*(this->groesseHashtabelle),cudaMemcpyHostToDevice);
           
        //Erstelle Ereignisse, um Dauer für GPU zu messen
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        //Suche nach einer Liste aller eingegebenen Zellen in der Hashtabelle
        dim3 threads(pGroesse);
        
        kernel_Dynamisch_Suchen<<<1,threads>>>(schluesselListe_GPU, hashtabelle_GPU,(this->groesseHashtabelle));
        
        //Kopiere Daten aus der GPU zur Hashtabelle
        cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Liste<T1,T2>)*(this->groesseHashtabelle), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
  
        float Millisekunden = 0;
        cudaEventElapsedTime(&Millisekunden, start, stop);
        float Sekunden = Millisekunden / 1000.0f;

        std::cout << "GPU sucht nach " << pGroesse << " Datenelementen in der Hashtabelle in ";
        std::cout << Millisekunden << " Millisekunden (" << (pGroesse / (double) Sekunden / 1000000.0f);
        std::cout << " Millionen Schlüssel pro Sekunde) bei quadratischen Hashverfahren." << std::endl;
        
        cudaFree(hashtabelle_GPU);
        cudaFree(schluesselListe_GPU);
        cudaFree(pSchluesselListe);

    }else{
        //TODO
        return;
    }
};

#endif