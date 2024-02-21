#ifndef HASHTABELLE_H
#define HASHTABELLE_H

#include <iostream>
#include <string>

#include <../data/datenvorlage.h>

template <typename T1, typename T2>
class Hashtabelle{
    private:
        size_t groesseHashtabelle;
        Zelle<T1,T2> * hashtabelle;
        hashtyp hashtyp_kode;

        size_t getHashwert(T1 pSchluessel);
        size_t getHashwert2(T1 pSchluessel);
        
        size_t getQuadratisch_Sondierungswert(size_t pIndex);
        std::string getZelle(size_t pIndex);
        
    public:
        Hashtabelle();
        Hashtabelle(hashtyp pHashtyp, size_t pGroesse);
        ~Hashtabelle();
        
        size_t getzahlZellen();
        size_t getGroesseHashtabelle();
        Zelle<T1,T2> * getHashtabelle(); 
    
        hashtyp getHashTyp();
        
        void drucken();

        void insert(T1 pSchluessel, T2 pWert);
        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, const size_t nx, const size_t ny, const size_t nz);

        bool suchen(T1 pSchluessel);
        void suchen_List(T1 * pSchluesselListe, const size_t nx, const size_t ny, const size_t nz);
};

#endif