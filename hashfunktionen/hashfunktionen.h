#ifndef HASHFUNKTIONEN_H
#define HASHFUNKTIONEN_H

#include <iostream>
#include <stdint.h>
#include <cmath>

namespace Hashfunktionen{
    //Berechne den Hashwert eines Schlüssels durch Modulo-Funktion
    template <typename T>
    size_t modulo_hash(T pSchluessel, size_t pGroesseHashtabelle){
        size_t schluessel_hash = (size_t) pSchluessel;
        return schluessel_hash%pGroesseHashtabelle;
    };
    
    //Berechne den Hashwert eines Schlüssels durch multiplikative Methode
    template <typename T>
    size_t multiplikativ_hash(T pSchluessel, size_t pGroesseHashtabelle){        
        double goldener_Schnitt, schluessel_hash, groesseHashtabelle_db;

        goldener_Schnitt = (sqrt(5.0)-1.0)/2.0;
        schluessel_hash = (double) pSchluessel;
        groesseHashtabelle_db = (double) pGroesseHashtabelle;

        schluessel_hash = floor(groesseHashtabelle_db*((schluessel_hash*goldener_Schnitt) - floor(schluessel_hash*goldener_Schnitt)));
    
        return schluessel_hash;
    };
    
    //Berechne den Hashwert eines Schlüssels durch perfekte Hashverfahren
    template <typename T, size_t a, size_t b, size_t p>
    size_t perfekt_hash(T pSchluessel){
        size_t schluessel_hash = (size_t) pSchluessel;
        return ((a*schluessel_hash + b)%p);
    };
    
    //Berechne den Hashwert eines Schlüssels durch Murmer-Hash
    template <typename T>
    size_t murmer_hash(T pSchluessel, size_t pGroesseHashtabelle){
        size_t schluessel_hash = (size_t) pSchluessel;
        schluessel_hash ^= schluessel_hash >> 16;
        schluessel_hash *= 0x85ebca6b;
        schluessel_hash ^= schluessel_hash >> 13;
        schluessel_hash *= 0xc2b2ae35;
        schluessel_hash ^= schluessel_hash >> 16;
        
        return (schluessel_hash & (pGroesseHashtabelle-1))%pGroesseHashtabelle;
    };
};

#endif