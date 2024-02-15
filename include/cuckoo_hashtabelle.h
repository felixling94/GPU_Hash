#ifndef CUCKOO_HASHTABELLE_H
#define CUCKOO_HASHTABELLE_H

#include <iostream>
#include <string>

#include <../include/hashfunktionen.h>
#include <../include/hashtabelle.h>

enum hashtabelle_nr{hashtabelle_1=1,hashtabelle_2=2};

template <typename T1, typename T2>
class Cuckoo_Hashtabelle{
    private:
        size_t groesseHashtabelle;

        Zelle<T1,T2> * hashtabelle1;
        hashfunktion hashfunktion_kode1;
        hashtyp hashtyp_kode1;

        Zelle<T1,T2> * hashtabelle2;
        hashfunktion hashfunktion_kode2;
        hashtyp hashtyp_kode2;

        size_t getHashwert(T1 pSchluessel, hashtabelle_nr pHashtabelle_nr);
        size_t getHashfunktion(T1 pSchluessel, hashfunktion pHashfunktion);
        size_t getQuadratisch_Sondierungswert(size_t pIndex);

        std::string getZelle(size_t pIndex, hashtabelle_nr pHashtabelle_nr);

        Zelle<T1,T2> vertauschen(hashtabelle_nr pHashtabelle_nr, size_t pIndex, Zelle<T1,T2> zelle_eingabe);

    public:
        Cuckoo_Hashtabelle();
        Cuckoo_Hashtabelle(hashtyp pHashtyp1, hashfunktion pHashfunktion1, hashtyp pHashtyp2, hashfunktion pHashfunktion2, size_t pGroesse);
        ~Cuckoo_Hashtabelle();
        
        size_t getGroesseHashtabelle();
        hashtyp getHashTyp(hashtabelle_nr pHashtabelle_nr);
        Zelle<T1,T2> * getHashtabelle(hashtabelle_nr pHashtabelle_nr); 
                   
        void drucken();
        
        void insert(T1 pSchluessel, T2 pWert);
        bool suchen(T1 pSchluessel);

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);

};

#endif