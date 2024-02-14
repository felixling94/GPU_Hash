#ifndef HASHFUNKTIONEN_H
#define HASHFUNKTIONEN_H

#include <iostream>
#include <stdint.h>

enum hashfunktion{modulo=0, murmer};

namespace Hashfunktionen{
    //Berechne den Hashwert eines Schlüssels durch Modulo-Funktion
    template <typename T1>
    size_t modulo_hash(T1 pSchluessel, size_t pGroesseHashtabelle){
        size_t schluessel_hash;
        schluessel_hash = (size_t) pSchluessel;

        return schluessel_hash%pGroesseHashtabelle;
    };

    //Berechne den Hashwert eines Schlüssels durch Murmer-Hash
    template <typename T1>
    size_t murmer_hash(T1 pSchluessel, size_t pGroesseHashtabelle){
        size_t schluessel_hash;
        
        schluessel_hash = (size_t) pSchluessel;
        schluessel_hash ^= schluessel_hash >> 16;
        schluessel_hash *= 0x85ebca6b;
        schluessel_hash ^= schluessel_hash >> 13;
        schluessel_hash *= 0xc2b2ae35;
        schluessel_hash ^= schluessel_hash >> 16;
        
        return (schluessel_hash & (pGroesseHashtabelle-1))%pGroesseHashtabelle;
    };
};

#endif