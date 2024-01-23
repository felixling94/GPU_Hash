#ifndef _HASHTABELLE_H_
#define _HASHTABELLE_H_

#include <stdint.h>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <hashtabelle.h>
#include <deklaration.cuh>

//Berechne den Index einer Zelle durch Modulo-Funktion
template <typename T0, typename T1>
DEVICEQUALIFIER T0 hashwert(T1 pSchluessel, T0 pGroesseHashtabelle){
    return (T0) pSchluessel % pGroesseHashtabelle;
}

//Berechne den Index einer Zelle durch Modulo-Funktion
template <typename T0>
DEVICEQUALIFIER T0 neuhashwert(T0 pIndex){
    T0 i = (T0) pow(ceil((double)pIndex/2),2.0);
    T0 j = (T0) pow(-1.0,(double)pIndex);
    return (i * j);
}

//Berechne den Index einer Zelle durch lineare Hashverfahren
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void hashLinear(Knoten<T1,T2> pKnoten, unsigned int pThreadid,
Knoten<T1,T2>* pHashtabelle, T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision, 
bool Cuckoo_stand){
    T0 i, neuIndex, pZahlKollision;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    pZahlKollision = 0;
    schluessel = pKnoten.schluessel;
    wert = pKnoten.wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    while(i < pGroesseHashtabelle){
        neuIndex = hashwert<T0,T1>(neuIndex + i,pGroesseHashtabelle);

        T1 vorher_schluessel = atomicCAS(&pHashtabelle[neuIndex].schluessel, 0, schluessel);

        if (vorher_schluessel == 0 || vorher_schluessel == schluessel){
            pHashtabelle[neuIndex].schluessel = schluessel;
            pHashtabelle[neuIndex].wert = wert;
            
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }

        ++i;
        ++pZahlKollision;
        
        if(Cuckoo_stand==true) {
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }
    }
}

//Berechne den Index einer Zelle durch quadratische Hashverfahren
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void hashQuadratisch(Knoten<T1,T2> pKnoten, unsigned int pThreadid,
Knoten<T1,T2>* pHashtabelle, T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision, 
bool Cuckoo_stand){
    T0 i, neuIndex, neuIndex2, pZahlKollision;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    pZahlKollision = 0;
    schluessel = pKnoten.schluessel;
    wert = pKnoten.wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);
    
    while (i/2<pGroesseHashtabelle){
        neuIndex2 = neuhashwert<T0>(i);
        neuIndex = hashwert<T0,T1>(neuIndex + neuIndex2,pGroesseHashtabelle);
        
        T1 vorher_schluessel = atomicCAS(&pHashtabelle[neuIndex].schluessel, 0, schluessel);
        
        if (vorher_schluessel == 0 || vorher_schluessel == schluessel){
            pHashtabelle[neuIndex].schluessel = schluessel;
            pHashtabelle[neuIndex].wert = wert;

            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }

        ++i;
        ++pZahlKollision;
        
        if(Cuckoo_stand==true) {
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }
    }
}

//Fuege der Hashtabelle eine Liste von Datenelemente durch lineare Hashverfahren hinzu.
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void insert_Linearhash(Knoten<T0,T1>* pKnoten, Knoten<T0,T1>* pHashtabelle,
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision){
    T0 i, threadid, neuIndex;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    hashLinear<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int)threadid,pHashtabelle,pGroesseHashtabelle,pKollision,false);
}

//Fuege der Hashtabelle eine Liste von Datenelemente durch quadratische Hashverfahren hinzu.
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void insert_Quadratischhash(Knoten<T1,T2>* pKnoten, Knoten<T1,T2>* pHashtabelle,
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision){
    T0 i, threadid, neuIndex, neuIndex2;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    hashQuadratisch<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int)threadid,pHashtabelle,pGroesseHashtabelle,pKollision,false);
}

//Fuege der Hashtabelle eine Liste von Datenelemente durch quadratische Hashverfahren hinzu.
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void insert_Cuckoohash(Knoten<T1,T2>* pKnoten, Knoten<T1,T2>* pHashtabelle1,
T0 pGroesseHashtabelle1, Knoten<T1,T2>* pHashtabelle2, T0 pGroesseHashtabelle2, 
Kollision<T0,T1,T2> * pKollision){
    T0 i,j, threadid, neuIndex1;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    j = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    
    neuIndex1 = hashwert<T0,T1>(schluessel,pGroesseHashtabelle1);
 
    //Lineare Hashverfahren und Quadratische Hashverfahren
    while(i<pGroesseHashtabelle1 || j<pGroesseHashtabelle2){
        if (i<pGroesseHashtabelle1)
            hashLinear<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int) threadid,pHashtabelle1,pGroesseHashtabelle1,pKollision,true);
        
        ++i;
        
        if (j<pGroesseHashtabelle2)
            hashQuadratisch<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int) threadid,pHashtabelle2,pGroesseHashtabelle2,pKollision,true);
        
        ++j;
    }
}


//Berechne den Index einer Zelle durch lineare Hashverfahren
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void hashLinear_suchen(Knoten<T1,T2> pKnoten,unsigned int pThreadid, Knoten<T1,T2>* pHashtabelle,
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision, bool Cuckoo_stand){
    T0 i, neuIndex, pZahlKollision;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    pZahlKollision = 0;
    schluessel = pKnoten.schluessel;
    wert = pKnoten.wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    while(i < pGroesseHashtabelle){
        neuIndex = hashwert<T0,T1>(neuIndex + i,pGroesseHashtabelle);
        T1 vorher_schluessel = pHashtabelle[neuIndex].schluessel;

        if (vorher_schluessel==schluessel){
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }

        ++i;
        ++pZahlKollision;
        
        if(Cuckoo_stand==true) {
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }
    }
}

//Berechne den Index einer Zelle durch quadratische Hashverfahren
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void hashQuadratisch_suchen(Knoten<T1,T2> pKnoten,unsigned int pThreadid, Knoten<T1,T2>* pHashtabelle,
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision, bool Cuckoo_stand){
    T0 i, neuIndex, neuIndex2, pZahlKollision;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    pZahlKollision = 0;
    schluessel = pKnoten.schluessel;
    wert = pKnoten.wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    while(i/2<pGroesseHashtabelle){
        neuIndex2 = neuhashwert<T0>(i);
        neuIndex = hashwert<T0,T1>(neuIndex + neuIndex2,pGroesseHashtabelle);

        T1 vorher_schluessel = pHashtabelle[neuIndex].schluessel;
    
        if (vorher_schluessel==schluessel){
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }
        
        ++i;
        ++pZahlKollision;
        
        if(Cuckoo_stand==true) {
            pKollision[pThreadid].knoten.schluessel = schluessel;
            pKollision[pThreadid].knoten.wert = wert;
            pKollision[pThreadid].zahlKollision = pZahlKollision;
            return;
        }
    }
}

//Suche nach einer Liste von Datenelementen in der Hashtabelle durch CUDA und Cuckoo-Hashverfahren
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void suchen_Cuckoohash(Knoten<T1,T2>* pKnoten, Knoten<T1,T2>* pHashtabelle1,
T0 pGroesseHashtabelle1, Knoten<T1,T2>* pHashtabelle2, T0 pGroesseHashtabelle2, Kollision<T0,T1,T2> * pKollision){
    T0 i,j, threadid, neuIndex1;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    j = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    
    neuIndex1 = hashwert<T0,T1>(schluessel,pGroesseHashtabelle1);
 
    //Lineare Hashverfahren und Quadratische Hashverfahren
    while(i<pGroesseHashtabelle1 || j<pGroesseHashtabelle2){
        if (i<pGroesseHashtabelle1)
            hashLinear_suchen<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int) threadid,pHashtabelle1,pGroesseHashtabelle1,pKollision,true);
       
        ++i;
        
        if (j<pGroesseHashtabelle2)
            hashQuadratisch_suchen<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int) threadid,pHashtabelle2,pGroesseHashtabelle2,pKollision,true);
    
        ++j;
    }
}


//Suche nach einem Datenelement in der Hashtabelle durch lineare Hashverfahren.
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void suchen_Linearhash(Knoten<T0,T1>* pKnoten, Knoten<T0,T1>* pHashtabelle, 
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision){
    T0 i, threadid, neuIndex;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    hashLinear_suchen<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int)threadid,pHashtabelle, pGroesseHashtabelle, pKollision, false);
}

//Suche nach einem Datenelement in der Hashtabelle durch quadratische Hashverfahren.
template <typename T0, typename T1, typename T2>
DEVICEQUALIFIER void suchen_Quadratischhash(Knoten<T0,T1>* pKnoten, Knoten<T0,T1>* pHashtabelle, 
T0 pGroesseHashtabelle, Kollision<T0,T1,T2> * pKollision){
    T0 i, threadid, neuIndex;
    T1 schluessel;
    T2 wert;
    
    i = 0;
    threadid = threadIdx.x;
    schluessel = pKnoten[threadid].schluessel;
    wert = pKnoten[threadid].wert;
    neuIndex = hashwert<T0,T1>(schluessel,pGroesseHashtabelle);

    hashQuadratisch_suchen<T0,T1,T2>(Knoten<T1,T2>{schluessel,wert},(unsigned int)threadid,pHashtabelle, pGroesseHashtabelle, pKollision, false);
}

#endif