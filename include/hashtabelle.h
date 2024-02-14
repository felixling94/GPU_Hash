#ifndef HASHTABELLE_H
#define HASHTABELLE_H

#include <iostream>
#include <string>

#include <../include/hashfunktionen.h>

#define FeldLeer 0xffffffff

//Arten der offenen Hashverfahren
//0: Keine Kollisionsauflösung
//1: Lineare Hashverfahren
//2: Quadratische Hashverfahren
//3: Beliebige Hashverfahren
enum hashtyp{keine_aufloesung=0, linear_aufloesung, quadratisch_aufloesung, 
             cuckoo_aufloesung, beliebig_aufloesung};

template <typename T1, typename T2>
struct Zelle{
    T1 schluessel = FeldLeer;
    T2 wert = FeldLeer;
};

template <typename T1, typename T2>
class Hashtabelle{
    private:
        size_t groesseHashtabelle;
        hashtyp hashtyp_kode;
        hashfunktion hashfunktion_kode;
        Zelle<T1,T2> * hashtabelle;

        size_t getHashwert(T1 pSchluessel);
        size_t getQuadratisch_Sondierungswert(size_t pIndex);

        std::string getZelle(size_t pIndex);

    public:
        Hashtabelle();
        Hashtabelle(hashtyp pHashtyp, hashfunktion pHashfunktion, size_t pGroesse);
        ~Hashtabelle();
        
        size_t getGroesseHashtabelle();
        hashtyp getHashTyp();
        hashfunktion getHashfunktion();
        Zelle<T1,T2> * getHashtabelle(); 
           
        void drucken();
        
        void insert(T1 pSchluessel, T2 pWert);
        bool suchen(T1 pSchluessel);

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);
};

#endif