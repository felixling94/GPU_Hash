#ifndef HASHTABELLE_DEVICE_CUH
#define HASHTABELLE_DEVICE_CUH

#include <stdint.h>

#include <../include/hashtabelle.h>
#include <../include/deklaration.cuh>
#include <../include/hashfunktionen.h>
#include <../include/hashfunktionen.cuh>

//Berechne den Index einer Zelle durch 32-Bit Murmur3-Hashfunktion
template <typename T1>
DEVICEQUALIFIER INLINEQUALIFIER size_t getHashwert(T1 pSchluessel, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    if (pHashfunktion == modulo){
        return modulo_hash<T1>(pSchluessel,pGroesseHashtabelle);
    }else if (pHashfunktion == murmer){
        return murmer_hash<T1>(pSchluessel,pGroesseHashtabelle);
    }else{
        //TODO
        return 0;
    }
}

//Berechne einen Sondierungswert
DEVICEQUALIFIER INLINEQUALIFIER size_t getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
}

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten ohne Kollisionsauflösung hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t threadid, index_neu;
    T1 schluessel, vorher_schluessel;
    T2 wert;
  
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);
    
    vorher_schluessel = atomicCAS(&pHashtabelle[index_neu].schluessel, FeldLeer, schluessel);
        
    if (vorher_schluessel == FeldLeer|| vorher_schluessel == schluessel){
        pHashtabelle[index_neu].schluessel = schluessel;
        pHashtabelle[index_neu].wert = wert;
        return;
    }

    return;
}

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch linear Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void linear_insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t i, threadid, index_neu;
    T1 schluessel, vorher_schluessel;
    T2 wert;
  
    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);
    
    while(i < pGroesseHashtabelle){
        index_neu = (index_neu + i )%pGroesseHashtabelle;
        vorher_schluessel = atomicCAS(&pHashtabelle[index_neu].schluessel, FeldLeer, schluessel);
        
        if (vorher_schluessel == FeldLeer|| vorher_schluessel == schluessel){
            pHashtabelle[index_neu].schluessel = schluessel;
            pHashtabelle[index_neu].wert = wert;
            return;
        }
        ++i;
    }

    return;
}

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch quadratische Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void quadratisch_insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t i, threadid, index_neu;
    T1 schluessel, vorher_schluessel;
    T2 wert;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);

    while((i/2) < pGroesseHashtabelle){
        index_neu = (index_neu + getQuadratisch_Sondierungswert(i))%pGroesseHashtabelle;
        vorher_schluessel = atomicCAS(&pHashtabelle[index_neu].schluessel,FeldLeer,schluessel);
        
        if (vorher_schluessel == FeldLeer || vorher_schluessel == schluessel){
            pHashtabelle[index_neu].schluessel = schluessel;
            pHashtabelle[index_neu].wert = wert;
            return;
        }
        ++i;
    } 
    
    return;   
}

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch Cuckoo-Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void cuckoo_insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle1, hashfunktion pHashfunktion1, 
Zelle<T1,T2>* pHashtabelle2, hashfunktion pHashfunktion2, size_t pGroesseHashtabelle){
    T1 schluessel, schluessel_temp;
    T2 wert, wert_temp;
    size_t i, threadid, index_neu1, index_neu2;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;

    index_neu1 = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion1);
    index_neu2 = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion2);

    while (i<pGroesseHashtabelle){
        index_neu1 = (index_neu1 + i )%pGroesseHashtabelle;
        index_neu2 = (index_neu2 + i )%pGroesseHashtabelle; 

        schluessel_temp = atomicCAS(&pHashtabelle1[index_neu1].schluessel,FeldLeer,schluessel);
        
        if (schluessel_temp == FeldLeer || schluessel_temp == schluessel){
            pHashtabelle1[index_neu1].schluessel = schluessel;
            pHashtabelle1[index_neu1].wert = wert;
            return;
        }

        schluessel_temp = pHashtabelle1[index_neu1].schluessel;
        wert_temp = pHashtabelle1[index_neu1].wert;

        pHashtabelle1[index_neu1].schluessel = schluessel;
        pHashtabelle1[index_neu1].wert = wert;

        schluessel = schluessel_temp;
        wert = wert_temp;

        schluessel_temp = atomicCAS(&pHashtabelle2[index_neu2].schluessel,FeldLeer,schluessel);
        
        if (schluessel_temp == FeldLeer || schluessel_temp == schluessel){
            pHashtabelle2[index_neu2].schluessel = schluessel;
            pHashtabelle2[index_neu2].wert = wert;
            return;
        }

        schluessel_temp = pHashtabelle2[index_neu2].schluessel;
        wert_temp = pHashtabelle2[index_neu2].wert;

        pHashtabelle2[index_neu2].schluessel = schluessel;
        pHashtabelle2[index_neu2].wert = wert;

        schluessel = schluessel_temp;
        wert = wert_temp;
        
        ++i;
    }
    return;
}

//TODO
//Beliebige Hashverfahren

//Suche nach einer Liste von Schlüsseln ohne Kollisionsauflösung
template <typename T1, typename T2>
DEVICEQUALIFIER void suchen(T1 * pSchluesselListe, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t threadid, index_neu;
    T1 schluessel, vorher_schluessel;

    threadid = threadIdx.x;
    schluessel = pSchluesselListe[threadid];
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);

    vorher_schluessel = pHashtabelle[index_neu].schluessel;

    if (vorher_schluessel==schluessel) return;

    return;
}

//Suche nach einer Liste von Schlüsseln durch lineare Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void linear_suchen(T1 * pSchluesselListe, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t i, threadid, index_neu;
    T1 schluessel, vorher_schluessel;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pSchluesselListe[threadid];
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);

    while(i < pGroesseHashtabelle ){
        index_neu = (index_neu + i )%pGroesseHashtabelle;
        vorher_schluessel = pHashtabelle[index_neu].schluessel;

        if (vorher_schluessel==schluessel) return;
        
        ++i; 
    }

    return;
}

//Suche nach einer Liste von Schlüsseln durch quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void quadratisch_suchen(T1 * pSchluesselListe, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    size_t i, threadid, index_neu; 
    T1 schluessel, vorher_schluessel;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pSchluesselListe[threadid];
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle,pHashfunktion);

    while((i/2) < pGroesseHashtabelle){
        index_neu = (index_neu + getQuadratisch_Sondierungswert(i))%pGroesseHashtabelle;
        vorher_schluessel = pHashtabelle[index_neu].schluessel;
    
        if (vorher_schluessel==schluessel) return;
        
        ++i;
    }  
    
    return;  
}

//TODO
//Cuckoo-Hashverfahren

//TODO
//Beliebige Hashverfahren

#endif