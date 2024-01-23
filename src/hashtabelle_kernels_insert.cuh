#ifndef HASHTABELLE_KERNELS_INSERT_CUH
#define HASHTABELLE_KERNELS_INSERT_CUH

#include <stdint.h>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/hashtabelle.h>
#include <../include/deklaration.cuh>
#include <../core/hashtabelle.cuh>

GLOBALQUALIFIER void kernel_hashinsert_Linear(Knoten<uint32_t,uint32_t>* pKnoten, 
Knoten<uint32_t,uint32_t>* pHashtabelle, uint32_t pGroesseHashtabelle, 
Kollision<uint32_t,uint32_t,uint32_t> * pKollision){
   insert_Linearhash<uint32_t,uint32_t,uint32_t>(pKnoten,pHashtabelle,pGroesseHashtabelle, pKollision);
}

GLOBALQUALIFIER void kernel_hashinsert_Quadratisch(Knoten<uint32_t,uint32_t>* pKnoten, 
Knoten<uint32_t,uint32_t>* pHashtabelle, uint32_t pGroesseHashtabelle, 
Kollision<uint32_t,uint32_t,uint32_t> * pKollision){
   insert_Quadratischhash<uint32_t,uint32_t,uint32_t>(pKnoten,pHashtabelle,pGroesseHashtabelle, pKollision);
}

GLOBALQUALIFIER void kernel_hashinsert_Cuckoo(Knoten<uint32_t,uint32_t>* pKnoten, 
Knoten<uint32_t,uint32_t>* pHashtabelle1,uint32_t pGroesseHashtabelle1, 
Knoten<uint32_t,uint32_t>* pHashtabelle2,uint32_t pGroesseHashtabelle2, 
Kollision<uint32_t,uint32_t,uint32_t> * pKollision){
  insert_Cuckoohash<uint32_t,uint32_t,uint32_t>(pKnoten,pHashtabelle1,pGroesseHashtabelle1,pHashtabelle2,pGroesseHashtabelle2, pKollision);
}

//Fuege der Hashtabelle ein Array von Schlüssel durch CUDA hinzu.
template <typename T0, typename T1, typename T2>
void Hashtabelle<T0,T1,T2>::insert_CUDA(Knoten<T1,T2> * pKnoten, size_t pKnotenGroesse, Kollision<T0,T1,T2> * pKollision){
   if (hashtyp_kode == keine_aufloesung){
        //TODO
   }else if (hashtyp_kode == linear_aufloesung){
      //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Knoten auf GPU
      Knoten<T1,T2> * hashtabelleGPU;
      Knoten<T1,T2> * knotenGPU;
      Kollision<T0,T1,T2> * KollisionGPU;
  
      cudaMalloc(&hashtabelleGPU,sizeof(Knoten<T1,T2>)*groesseHashtabelle);
      cudaMalloc(&knotenGPU,sizeof(Knoten<T1,T2>)*(size_t)pKnotenGroesse);
      cudaMalloc(&KollisionGPU,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse);

      cudaMemcpy(hashtabelleGPU,hashtabelle,sizeof(Knoten<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
      cudaMemcpy(knotenGPU,pKnoten,sizeof(Knoten<T1,T2>)* (size_t)pKnotenGroesse,cudaMemcpyHostToDevice);
      cudaMemcpy(KollisionGPU,pKollision,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse,cudaMemcpyHostToDevice);

      //Erstelle Ereignisse, um Dauer für GPU zu messen
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);

      //Fuege der Hashtabelle alle eingegebenen Knoten hinzu
      dim3 threads(pKnotenGroesse);

      kernel_hashinsert_Linear<<<1,threads>>>(knotenGPU, hashtabelleGPU, groesseHashtabelle,KollisionGPU);

      //Kopiere Daten aus der GPU zur Hashtabelle
      cudaMemcpy(hashtabelle, hashtabelleGPU, sizeof(Knoten<T1,T2>)*(size_t)groesseHashtabelle, cudaMemcpyDeviceToHost);
      cudaMemcpy(pKollision, KollisionGPU, sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse, cudaMemcpyDeviceToHost);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
  
      float Millisekunden = 0;
      cudaEventElapsedTime(&Millisekunden, start, stop);
      float Sekunden = Millisekunden / 1000.0f;

      std::cout << "GPU hat der Hashtabelle " << pKnotenGroesse << " in " << Millisekunden;
      std::cout << " Millisekunden (" << (pKnotenGroesse / (double) Sekunden / 1000000.0f);
      std::cout << " Millionen Schlüssel pro Sekunde) bei linearen Hashverfahren hinzugefügt.";
      std::cout << "" << std::endl;
        
      cudaFree(hashtabelleGPU);
      cudaFree(KollisionGPU);
      cudaFree(pKnoten);

   }else if (hashtyp_kode == quadratisch_aufloesung){
      //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Knoten auf GPU
      Knoten<T1,T2> * hashtabelleGPU;
      Knoten<T1,T2> * knotenGPU;
      Kollision<T0,T1,T2> * KollisionGPU;
      
      cudaMalloc(&hashtabelleGPU,sizeof(Knoten<T1,T2>)*groesseHashtabelle);
      cudaMalloc(&knotenGPU,sizeof(Knoten<T1,T2>)*(size_t)pKnotenGroesse);
      cudaMalloc(&KollisionGPU,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse);

      cudaMemcpy(hashtabelleGPU,hashtabelle,sizeof(Knoten<T1,T2>)*groesseHashtabelle,cudaMemcpyHostToDevice);
      cudaMemcpy(knotenGPU,pKnoten,sizeof(Knoten<T1,T2>)* (size_t)pKnotenGroesse,cudaMemcpyHostToDevice);
      cudaMemcpy(KollisionGPU,pKollision,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse,cudaMemcpyHostToDevice);
      
      //Erstelle Ereignisse, um Dauer für GPU zu messen
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);

      //Fuege der Hashtabelle alle eingegebenen Knoten hinzu
      dim3 threads(pKnotenGroesse);

      kernel_hashinsert_Quadratisch<<<1,threads>>>(knotenGPU, hashtabelleGPU, groesseHashtabelle,KollisionGPU);

      //Kopiere Daten aus der GPU zur Hashtabelle
      cudaMemcpy(hashtabelle, hashtabelleGPU, sizeof(Knoten<T1,T2>)*(size_t)groesseHashtabelle, cudaMemcpyDeviceToHost);
      cudaMemcpy(pKollision, KollisionGPU, sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse, cudaMemcpyDeviceToHost);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
  
      float Millisekunden = 0;
      cudaEventElapsedTime(&Millisekunden, start, stop);
      float Sekunden = Millisekunden / 1000.0f;

      std::cout << "GPU hat der Hashtabelle " << pKnotenGroesse << " in " << Millisekunden;
      std::cout << " Millisekunden (" << (pKnotenGroesse / (double) Sekunden / 1000000.0f);
      std::cout << " Millionen Schlüssel pro Sekunde) bei quadratischen Hashverfahren hinzugefügt.";
      std::cout << "" << std::endl;
        
      cudaFree(hashtabelleGPU);
      cudaFree(KollisionGPU);
      cudaFree(pKnoten);
   }
}

//Fuege der Hashtabelle ein Array von Schlüssel durch CUDA und Cuckoo-Hashverfahren hinzu.
template <typename T0, typename T1, typename T2>
void insert_Cuckoo_CUDA(Knoten<T1,T2> * pKnoten, size_t pKnotenGroesse,
Hashtabelle<T0,T1,T2> pHashtabelle1, Hashtabelle<T0,T1,T2> pHashtabelle2,
Kollision<T0,T1,T2> * pKollision){
   //Hole Daten aus zwei Hashtabelle heraus.
   Knoten<T1,T2> * usrHashtabelle1 = pHashtabelle1.getHashtabelle();
   Knoten<T1,T2> * usrHashtabelle2 = pHashtabelle2.getHashtabelle();

   T0 usrGroesseHashtabelle1 = pHashtabelle1.getGroesseHashtabelle();
   T0 usrGroesseHashtabelle2 = pHashtabelle2.getGroesseHashtabelle();

   //Reserviere und kopiere Daten aus der Hashtabelle und eingegebenen Knoten auf GPU
   Knoten<T1,T2> * hashtabelleGPU1;
   Knoten<T1,T2> * hashtabelleGPU2;
   Knoten<T1,T2> * knotenGPU;
   Kollision<T0,T1,T2> * KollisionGPU;
   
   cudaMalloc(&hashtabelleGPU1,sizeof(Knoten<T1,T2>)*(size_t)usrGroesseHashtabelle1);
   cudaMalloc(&hashtabelleGPU2,sizeof(Knoten<T1,T2>)*(size_t)usrGroesseHashtabelle2);
   cudaMalloc(&knotenGPU,sizeof(Knoten<T1,T2>)*(size_t)pKnotenGroesse);
   cudaMalloc(&KollisionGPU,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse);
    
   cudaMemcpy(hashtabelleGPU1,usrHashtabelle1,sizeof(Knoten<T1,T2>)*usrGroesseHashtabelle1,cudaMemcpyHostToDevice);
   cudaMemcpy(hashtabelleGPU2,usrHashtabelle2,sizeof(Knoten<T1,T2>)*usrGroesseHashtabelle2,cudaMemcpyHostToDevice);
   cudaMemcpy(knotenGPU,pKnoten,sizeof(Knoten<T1,T2>)* (size_t)pKnotenGroesse,cudaMemcpyHostToDevice);
   cudaMemcpy(KollisionGPU,pKollision,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse,cudaMemcpyHostToDevice);
  
   //Erstelle Ereignisse, um Dauer für GPU zu messen
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);

   //Fuege der Hashtabelle alle eingegebenen Knoten hinzu
   dim3 threads(pKnotenGroesse);

   kernel_hashinsert_Cuckoo<<<1,threads>>>(knotenGPU,hashtabelleGPU1,usrGroesseHashtabelle1,hashtabelleGPU2,usrGroesseHashtabelle2,KollisionGPU);
   
   //Kopiere Daten aus der GPU zur Hashtabelle
   cudaMemcpy(usrHashtabelle1, hashtabelleGPU1, sizeof(Knoten<T1,T2>)*(size_t)usrGroesseHashtabelle1, cudaMemcpyDeviceToHost);
   cudaMemcpy(usrHashtabelle2, hashtabelleGPU2, sizeof(Knoten<T1,T2>)*(size_t)usrGroesseHashtabelle2, cudaMemcpyDeviceToHost);  
   cudaMemcpy(pKollision,KollisionGPU,sizeof(Kollision<T0,T1,T2>)*pKnotenGroesse,cudaMemcpyDeviceToHost);
  
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
  
   float Millisekunden = 0;
   cudaEventElapsedTime(&Millisekunden, start, stop);
   float Sekunden = Millisekunden / 1000.0f;

   std::cout << "GPU hat der Hashtabelle " << pKnotenGroesse << " in " << Millisekunden;
   std::cout << " Millisekunden (" << (pKnotenGroesse / (double) Sekunden / 1000000.0f);
   std::cout << " Millionen Schlüssel pro Sekunde) bei Cuckoo Hashverfahren hinzugefügt.";
   std::cout << "" << std::endl;
      
   cudaFree(hashtabelleGPU1);
   cudaFree(hashtabelleGPU2);
   cudaFree(KollisionGPU);
   cudaFree(pKnoten);
}

#endif