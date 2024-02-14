#ifndef HASHTABELLE_CUH
#define HASHTABELLE_CUH

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/hashtabelle.h>
#include <../core/hashtabelle_kernels.cuh>

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse){
    Zelle<T1,T2> * zellen_neu;

    Zelle<T1,T2> * hashtabelle_GPU;
    Zelle<T1,T2> * zellen_GPU;

    if(pGroesse <= groesseHashtabelle){
        std::vector<Zelle<T1,T2>> schluesselwertVektor;
        schluesselwertVektor.reserve(pGroesse);
        
        for (size_t i = 0; i < pGroesse ; i++)
            schluesselwertVektor.push_back(Zelle<T1,T2>{pSchluesselListe[i],pWerteListe[i]});
        
        zellen_neu = schluesselwertVektor.data();

        if (hashtyp_kode == keine_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&zellen_GPU,sizeof(Zelle<T1,T2>)*pGroesse);
            
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(zellen_GPU,zellen_neu,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice);

            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            dim3 threads(pGroesse);

            kernel_Insert<<<1,threads>>>(zellen_GPU, hashtabelle_GPU, groesseHashtabelle,hashfunktion_kode);

            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;

            std::cout << "Zahl der zu belegenden Zellen (in Datenelementen)                 : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der belegten Zellen pro Sekunde (in Millionen Datenelementen   ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;
        
            cudaFree(hashtabelle_GPU);
            cudaFree(zellen_GPU);
            cudaFree(zellen_neu);

            return;
        }else if (hashtyp_kode == linear_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&zellen_GPU,sizeof(Zelle<T1,T2>)*pGroesse);
            
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(zellen_GPU,zellen_neu,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice);

            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            dim3 threads(pGroesse);

            kernel_Linear_Insert<<<1,threads>>>(zellen_GPU, hashtabelle_GPU, groesseHashtabelle,hashfunktion_kode);

            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;

            std::cout << "Zahl der zu belegenden Zellen (in Datenelementen)                 : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der belegten Zellen pro Sekunde (in Millionen Datenelementen   ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;
        
            cudaFree(hashtabelle_GPU);
            cudaFree(zellen_GPU);
            cudaFree(zellen_neu);

            return;
        }else if (hashtyp_kode == quadratisch_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&zellen_GPU,sizeof(Zelle<T1,T2>)*pGroesse);
            
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(zellen_GPU,zellen_neu,sizeof(Zelle<T1,T2>)*pGroesse,cudaMemcpyHostToDevice);
 
            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Fuege der Hashtabelle alle eingegebenen Zellen hinzu
            dim3 threads(pGroesse);

            kernel_Quadratisch_Insert<<<1,threads>>>(zellen_GPU, hashtabelle_GPU, groesseHashtabelle,hashfunktion_kode);
        
            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;

            std::cout << "Zahl der belegenden Zellen (in Datenelementen)                    : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der belegten Zellen pro Sekunde (in Millionen Datenelementen   ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;
        
            cudaFree(hashtabelle_GPU);
            cudaFree(zellen_GPU);
            cudaFree(zellen_neu);

            return;
        }else{
        //TODO
            return;
        }

    }else{
        std::cout << "Die Größe der der Hashtabelle hinzufügenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
        return;
    }
};

//Suche nach einem Array von Schlüsseln in der Hashtabelle gleichzeitig.
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::suchen_List(T1 * pSchluesselListe, size_t pGroesse){
    Zelle<T1,T2> * hashtabelle_GPU;
    T1 * schluesselListe_GPU;

    if(pGroesse <= groesseHashtabelle){
        if (hashtyp_kode == keine_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle, eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&schluesselListe_GPU,sizeof(T1)*pGroesse);
        
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(schluesselListe_GPU,pSchluesselListe,sizeof(T1)* pGroesse,cudaMemcpyHostToDevice);

            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Suche nach einer Liste aller eingegebenen Zellen in der Hashtabelle
            dim3 threads(pGroesse);
        
            kernel_Suchen<<<1,threads>>>(schluesselListe_GPU, hashtabelle_GPU,groesseHashtabelle,hashfunktion_kode);
        
            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;
            
            std::cout << "Zahl der gesuchten Zellen in der Hashtabelle (in Datenelementen)  : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der gesuchten Zellen pro Sekunde (in Millionen Datenelementen  ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;

            cudaFree(hashtabelle_GPU);
            cudaFree(schluesselListe_GPU);

        }else if (hashtyp_kode == linear_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle, eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&schluesselListe_GPU,sizeof(T1)*pGroesse);
        
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(schluesselListe_GPU,pSchluesselListe,sizeof(T1)* pGroesse,cudaMemcpyHostToDevice);

            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Suche nach einer Liste aller eingegebenen Zellen in der Hashtabelle
            dim3 threads(pGroesse);
        
            kernel_Linear_Suchen<<<1,threads>>>(schluesselListe_GPU, hashtabelle_GPU,groesseHashtabelle,hashfunktion_kode);
        
            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;
            
            std::cout << "Zahl der gesuchten Zellen in der Hashtabelle (in Datenelementen)  : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der gesuchten Zellen pro Sekunde (in Millionen Datenelementen  ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;

            cudaFree(hashtabelle_GPU);
            cudaFree(schluesselListe_GPU);

        }else if (hashtyp_kode == quadratisch_aufloesung){
            //Reserviere und kopiere Daten aus der Hashtabelle, eingegebenen Zellen auf GPU
            cudaMalloc(&hashtabelle_GPU,sizeof(Zelle<T1,T2>)*groesseHashtabelle);
            cudaMalloc(&schluesselListe_GPU,sizeof(T1)*pGroesse);
        
            cudaMemcpy(hashtabelle_GPU,hashtabelle,sizeof(Zelle<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
            cudaMemcpy(schluesselListe_GPU,pSchluesselListe,sizeof(T1)* pGroesse,cudaMemcpyHostToDevice);

            //Erstelle Ereignisse, um Dauer für GPU zu messen
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            //Suche nach einer Liste aller eingegebenen Zellen in der Hashtabelle
            dim3 threads(pGroesse);
        
            kernel_Quadratisch_Suchen<<<1,threads>>>(schluesselListe_GPU, hashtabelle_GPU,groesseHashtabelle,hashfunktion_kode);
        
            //Kopiere Daten aus der GPU zur Hashtabelle
            cudaMemcpy(hashtabelle, hashtabelle_GPU, sizeof(Zelle<T1,T2>)*groesseHashtabelle, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
            float Millisekunden = 0;
            cudaEventElapsedTime(&Millisekunden, start, stop);
            float Sekunden = Millisekunden / 1000.0f;

            std::cout << "Zahl der gesuchten Zellen in der Hashtabelle (in Datenelementen)  : ";
            std::cout << pGroesse << std::endl;
            std::cout << "Dauer (in Millisekunden)                                          : ";
            std::cout <<  Millisekunden << std::endl;
            std::cout << "Zahl der gesuchten Zellen pro Sekunde (in Millionen Datenelementen  ";
            std::cout << std::endl;
            std::cout << "pro Sekunde)                                                      : ";
            std::cout << (pGroesse / (double) Sekunden / 1000000.0f) << std::endl;
            std::cout << std::endl;
        
            cudaFree(hashtabelle_GPU);
            cudaFree(schluesselListe_GPU);
        
        }else{
            //TODO
        }

    }else{
        std::cout << "Die Größe der der Hashtabelle nach suchenden Schlüssel muss mindestens 0 und höchstens ";
        std::cout << " die Größe der Hashtabelle betragen." << std::endl;
        return;
    }
};

#endif