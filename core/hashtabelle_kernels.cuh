#ifndef HASHTABELLE_KERNELS_CUH
#define HASHTABELLE_KERNELS_CUH

#include <stdint.h>

#include <../include/hashtabelle.h>
#include <../include/deklaration.cuh>
#include <../include/hashfunktionen.h>
#include <../core/hashtabelle_device.cuh>

/////////////////////////////////////////////////////////////////////////////////////////
//Speicherung von einer Liste von Paaren von Schlüsseln und Werten in der Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
GLOBALQUALIFIER void kernel_Insert(Zelle<uint32_t,uint32_t> * pZellen, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    insert<uint32_t,uint32_t>(pZellen,pHashtabelle,pGroesseHashtabelle, pHashfunktion);
}

GLOBALQUALIFIER void kernel_Linear_Insert(Zelle<uint32_t,uint32_t> * pZellen, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    linear_insert<uint32_t,uint32_t>(pZellen,pHashtabelle,pGroesseHashtabelle,pHashfunktion);
}

GLOBALQUALIFIER void kernel_Quadratisch_Insert(Zelle<uint32_t,uint32_t> * pZellen, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle,hashfunktion pHashfunktion){
    quadratisch_insert<uint32_t,uint32_t>(pZellen,pHashtabelle,pGroesseHashtabelle,pHashfunktion);
}

/////////////////////////////////////////////////////////////////////////////////////////
//Suche nach einer Liste von Paaren von Schlüsseln in der Hashtabelle
/////////////////////////////////////////////////////////////////////////////////////////
GLOBALQUALIFIER void kernel_Suchen(uint32_t * pSchluesselListe, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle,hashfunktion pHashfunktion){
    suchen<uint32_t,uint32_t>(pSchluesselListe,pHashtabelle,pGroesseHashtabelle,pHashfunktion);
}

GLOBALQUALIFIER void kernel_Linear_Suchen(uint32_t * pSchluesselListe, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle,hashfunktion pHashfunktion){
    linear_suchen<uint32_t,uint32_t>(pSchluesselListe,pHashtabelle,pGroesseHashtabelle,pHashfunktion);
}

GLOBALQUALIFIER void kernel_Quadratisch_Suchen(uint32_t * pSchluesselListe, 
Zelle<uint32_t,uint32_t> * pHashtabelle, size_t pGroesseHashtabelle,hashfunktion pHashfunktion){
    quadratisch_suchen<uint32_t,uint32_t>(pSchluesselListe,pHashtabelle,pGroesseHashtabelle,pHashfunktion);
}

#endif