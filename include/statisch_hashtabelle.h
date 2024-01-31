#ifndef STATISCH_HASHTABELLE_H
#define STATISCH_HASHTABELLE_H

#include <iostream>
#include <string>

#include <hashtabelle.h>

//Arten der offenen Hashverfahren
//0: Keine Kollisionsauflösung
//1: Lineare Hashverfahren
//2: Quadratische Hashverfahren
//3: Beliebige Hashverfahren
enum hashtyp{keine_aufloesung=0, linear_aufloesung, 
             quadratisch_aufloesung, beliebig_aufloesung};

template <typename T1, typename T2>
class Statisch_Hashtabelle: public Hashtabelle<T1>{
    private:
        hashtyp hashtyp_kode;
        Zelle<T1,T2> * hashtabelle;

        size_t getQuadratisch_Sondierungswert(size_t pIndex);

    public:
        Statisch_Hashtabelle();
        Statisch_Hashtabelle(hashtyp pHashtyp, size_t pGroesse);
        ~Statisch_Hashtabelle();

        Zelle<T1,T2> * getHashtabelle();
        std::string getZelle(size_t pIndex);
        void drucken();

        void insert(T1 pSchluessel, T2 pWert);
        bool suchen(T1 pSchluessel);

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);
};

#endif