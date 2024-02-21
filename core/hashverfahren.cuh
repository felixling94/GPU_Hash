#ifndef HASHTABELLE_DEVICE_CUH
#define HASHTABELLE_DEVICE_CUH

#include <stdint.h>

#include <../data/datenvorlage.h>
#include <../include/deklaration.cuh>
#include <../hashfunktionen/dycuckoo_funktionen.cuh>

//Berechne den Index einer Zelle durch DyCuckoo-Hash3-Funktion
template <typename T>
DEVICEQUALIFIER INLINEQUALIFIER size_t getHashwert(T pSchluessel, size_t pGroesseHashtabelle){
    return DyCuckoo_Funktionen_Device::hash3(pSchluessel)%pGroesseHashtabelle;
};

//Berechne den Index einer Zelle durch DyCuckoo-Hash5-Funktion
template <typename T>
DEVICEQUALIFIER INLINEQUALIFIER size_t getHashwert2(T pSchluessel, size_t pGroesseHashtabelle){
    return DyCuckoo_Funktionen_Device::hash3(pSchluessel)%pGroesseHashtabelle;
};

//Berechne einen Sondierungswert
DEVICEQUALIFIER INLINEQUALIFIER size_t getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
};

//Füge der Hashtabelle einen Schlüssel und dessen Wert ohne Kollisionsauflösung hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void insert0(T1 pSchluessel, T2 pWert, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t index_neu;
    T1 schluessel_neu;

    index_neu = getHashwert<T1>(pSchluessel,pGroesseHashtabelle);
    schluessel_neu = atomicCAS(&pHashtabelle[index_neu].schluessel, LeerFeld, pSchluessel);
    
    if (schluessel_neu==LeerFeld || schluessel_neu==pSchluessel){
        pHashtabelle[index_neu].schluessel = pSchluessel;
        pHashtabelle[index_neu].wert = pWert;
        __syncthreads();
        return;
    }   
};

//Füge der Hashtabelle einen Schlüssel und dessen Wert durch linear Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void insert1(T1 pSchluessel, T2 pWert, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, max_hashtabellegroesse, index_neu;
    T1 schluessel_neu;

    i = 0;
    max_hashtabellegroesse = (size_t)((100+PROZENT_SCHLEIFE)/100*pGroesseHashtabelle);
    index_neu = getHashwert<T1>(pSchluessel,pGroesseHashtabelle);
    
    while(i < max_hashtabellegroesse){
        index_neu = (index_neu+i)%pGroesseHashtabelle;
        schluessel_neu = atomicCAS(&pHashtabelle[index_neu].schluessel, LeerFeld, pSchluessel);
        
        if (schluessel_neu==LeerFeld || schluessel_neu==pSchluessel){
            pHashtabelle[index_neu].schluessel = pSchluessel;
            pHashtabelle[index_neu].wert = pWert;
            __syncthreads();
            break;
        }
        ++i;
    }

};

//Füge der Hashtabelle einen Schlüssel und dessen Wert durch quadratische Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void insert2(T1 pSchluessel, T2 pWert, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, max_hashtabellegroesse, index_neu;
    T1 schluessel_neu;

    i = 0;
    max_hashtabellegroesse = (size_t)((100+PROZENT_SCHLEIFE)/100*pGroesseHashtabelle);
    index_neu = getHashwert<T1>(pSchluessel,pGroesseHashtabelle);
    
    while((i/2) < max_hashtabellegroesse){
        index_neu = (index_neu+getQuadratisch_Sondierungswert(i))%pGroesseHashtabelle;
        schluessel_neu = atomicCAS(&pHashtabelle[index_neu].schluessel, LeerFeld, pSchluessel);
        
        if (schluessel_neu==LeerFeld || schluessel_neu==pSchluessel){
            pHashtabelle[index_neu].schluessel = pSchluessel;
            pHashtabelle[index_neu].wert = pWert;
            __syncthreads();
            break;
        }
        ++i;
    }

};

//Füge der Hashtabelle einen Schlüssel und dessen Wert durch doppelte Hashverfahren hinzu.
template <typename T1, typename T2>
DEVICEQUALIFIER void insert3(T1 pSchluessel, T2 pWert, Zelle<T1,T2>* pHashtabelle, size_t pGroesseHashtabelle){
    size_t i, max_hashtabellegroesse, index_neu;
    T1 schluessel_neu;
    
    i = 0;
    max_hashtabellegroesse = (size_t)((100+PROZENT_SCHLEIFE)/100*pGroesseHashtabelle);
    index_neu = getHashwert<T1>(pSchluessel,pGroesseHashtabelle);
    
    while(i < max_hashtabellegroesse){
        index_neu = (index_neu + i*getHashwert2(pSchluessel,pGroesseHashtabelle))%pGroesseHashtabelle;
        schluessel_neu = atomicCAS(&pHashtabelle[index_neu].schluessel, LeerFeld, pSchluessel);
        
        if (schluessel_neu==LeerFeld || schluessel_neu==pSchluessel){
            pHashtabelle[index_neu].schluessel = pSchluessel;
            pHashtabelle[index_neu].wert = pWert;
            __syncthreads();
            break;
        }
        ++i;
    }

};

//TODO

#endif