#ifndef DYNAMISCH_HASHTABELLE_H
#define DYNAMISCH_HASHTABELLE_H

#include <iostream>
#include <string>

#include <hashtabelle.h>

template <typename T1, typename T2>
struct Liste{
    Zelle<T1,T2> jetzt;
    Zelle<T1,T2> naechste;
};

template <typename T1, typename T2>
class Dynamisch_Hashtabelle: public Hashtabelle<T1>{
    private:
        bool standAuflosung;
        Liste<T1,T2> * hashtabelle;

    public:
        Dynamisch_Hashtabelle();
        Dynamisch_Hashtabelle(bool pStandAuflosung, size_t pGroesse);
        ~Dynamisch_Hashtabelle();

        Liste<T1,T2> * getHashtabelle();
        std::string getZelle(size_t pIndex);
        void drucken();

        void insert(T1 pSchluessel, T2 pWert);
        bool suchen(T1 pSchluessel);

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);
};

#endif