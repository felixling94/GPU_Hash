#ifndef _HASHTABELLE_CUH_
#define _HASHTABELLE_CUH_

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <../include/statisch_hashtabelle.h>
#include <../include/dynamisch_hashtabelle.h>
#include <../include/deklaration.cuh>

//Berechne den Index einer Zelle durch Modulo-Funktion
template <typename T1>
DEVICEQUALIFIER size_t getHashwert(T1 pSchluessel, size_t pGroesseHashtabelle){
    return (size_t)pSchluessel % pGroesseHashtabelle;
};

//Berechne einen Sondierungswert
DEVICEQUALIFIER size_t getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
};

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch linear Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Linear_Insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, threadid, index_neu;
    T1 schluessel;
    T2 wert;
  
    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle);
    
    while(i < pGroesseHashtabelle){
        index_neu = getHashwert<T1>(schluessel + (T1) i, pGroesseHashtabelle);
        
        T1 vorher_schluessel = atomicCAS(&pHashtabelle[index_neu].schluessel, 0, schluessel);
        
        if (vorher_schluessel == 0 || vorher_schluessel == schluessel){
            pHashtabelle[index_neu].schluessel = schluessel;
            pHashtabelle[index_neu].wert = wert;
            return;
        }

        ++i;
    }

    return;
};

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch quadratische Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Quadratisch_Insert(Zelle<T1,T2>* pZellen, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, threadid, index_neu, sondierungswert;
    T1 schluessel;
    T2 wert;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle);

    while((i/2) < pGroesseHashtabelle){
        sondierungswert = getQuadratisch_Sondierungswert(i);
        index_neu = getHashwert<T1>(index_neu + (T1) sondierungswert,pGroesseHashtabelle);
        
        T1 vorher_schluessel = atomicCAS(&pHashtabelle[index_neu].schluessel, 0, schluessel);
        
        if (vorher_schluessel == 0 || vorher_schluessel == schluessel){
            pHashtabelle[index_neu].schluessel = schluessel;
            pHashtabelle[index_neu].wert = wert;
            return;
        }

        ++i;
    } 

    return;   
};

//TODO
//Cuckoo-Hashverfahren

//TODO
//Beliebige Hashverfahren

//Füge der Hashtabelle eine Liste von Schlüsseln und deren Werten durch dynamische Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Dynamisch_Insert(Zelle<T1,T2>* pZellen, Liste<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, threadid, index_vorher, index_nachher;
    T1 schluessel;
    T2 wert;
  
    i = 0;
    threadid = threadIdx.x;
    schluessel = pZellen[threadid].schluessel;
    wert = pZellen[threadid].wert;
    index_vorher = getHashwert<T1>(schluessel,pGroesseHashtabelle);
    index_nachher = index_vorher;
    
    while(i < pGroesseHashtabelle){
        index_nachher+=i;

        T1 vorher_schluessel = atomicCAS(&pHashtabelle[index_nachher].jetzt.schluessel, 0, schluessel);
        
        if(vorher_schluessel == 0 || vorher_schluessel == schluessel){
            pHashtabelle[index_nachher].jetzt.schluessel = schluessel;
            pHashtabelle[index_nachher].jetzt.wert = wert;

            if (i>0){
                pHashtabelle[index_vorher].naechste.schluessel = schluessel;
                pHashtabelle[index_vorher].naechste.wert = wert;
            }

            return;
        }
        ++i;
    }

    return;
};

//Suche nach einer Liste von Schlüsseln durch lineare Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Linear_Suchen(T1 * pSchluesselListe, Zelle<T1,T2> * pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, threadid, index_neu;
    T1 schluessel;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pSchluesselListe[threadid];
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle);

    while(i < pGroesseHashtabelle ){
        index_neu = getHashwert<T1>(schluessel + (T1) i, pGroesseHashtabelle);

        T1 vorher_schluessel = pHashtabelle[index_neu].schluessel;

        if (vorher_schluessel==schluessel){
            //TODO
            return;
        }

        ++i;
    }

    return;
};

//Suche nach einer Liste von Schlüsseln durch quadratische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Quadratisch_Suchen(T1 * pSchluesselListe, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, threadid, index_neu, sondierungswert;
    T1 schluessel;

    i = 0;
    threadid = threadIdx.x;
    schluessel = pSchluesselListe[threadid];
    index_neu = getHashwert<T1>(schluessel,pGroesseHashtabelle);

    while((i/2) < pGroesseHashtabelle){
        sondierungswert = getQuadratisch_Sondierungswert(i);
        index_neu = getHashwert<T1>(index_neu + (T1) sondierungswert,pGroesseHashtabelle);
       
        T1 vorher_schluessel = pHashtabelle[index_neu].schluessel;
    
        if (vorher_schluessel==schluessel){
            //TODO
            return;
        }
        
        ++i;
    }  

    return;  
};

//TODO
//Cuckoo-Hashverfahren

//TODO
//Beliebige Hashverfahren

//Suche nach einer Liste von Schlüsseln durch dynamische Hashverfahren
template <typename T1, typename T2>
DEVICEQUALIFIER void hashtabelle_Dynamisch_Suchen(T1 * pSchluesselListe, Liste<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    //TODO
};

#endif