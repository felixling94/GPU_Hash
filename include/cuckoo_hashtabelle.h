#ifndef CUCKOO_HASHTABELLE_H
#define CUCKOO_HASHTABELLE_H

#include <iostream>
#include <string>

#include <../include/hashfunktionen.h>
#include <../include/hashtabelle.h>

template <typename T1, typename T2>
class Cuckoo_Hashtabelle{
    private:
        size_t groesseHashtabelle;

        Zelle<T1,T2> * hashtabelle1;
        hashfunktion hashfunktion1;
        hashtyp hashtyp1;

        Zelle<T1,T2> * hashtabelle2;
        hashfunktion hashfunktion2;
        hashtyp hashtyp2;

        size_t getHashwert(T1 pSchluessel, hashfunktion pHashfunktion);
 
    public:
        Cuckoo_Hashtabelle();
        Cuckoo_Hashtabelle(hashtyp pHashtyp1, hashfunktion pHashfunktion1, hashtyp pHashtyp2, hashfunktion pHashfunktion2, size_t pGroesse);
        ~Cuckoo_Hashtabelle();
        
        void drucken();

        void insert(T1 pSchluessel, T2 pWert);
        bool suchen(T1 pSchluessel);

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);
};

#endif