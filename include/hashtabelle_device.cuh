#ifndef HASHTABELLE_DEVICE_H
#define HASHTABELLE_DEVICE_H

#include <iostream>
#include <string>

#include <../data/datenvorlage.h>

template <typename T1, typename T2>
class Hashtabelle_Device{
    private:
        size_t groesseHashtabelle;
        Zelle<T1,T2> * hashtabelle;
        hashtyp hashtyp_kode;

        std::string getZelle(size_t pIndex);

        void insert0(Zelle<T1,T2> * zellen, size_t pGroesse);
        void insert_linear(Zelle<T1,T2> * zellen, size_t pGroesse);
        void insert_quadratisch(Zelle<T1,T2> * zellen, size_t pGroesse);
        void insert_doppelt(Zelle<T1,T2> * zellen, size_t pGroesse);
        
    public:
        Hashtabelle_Device();
        Hashtabelle_Device(hashtyp pHashtyp, size_t pGroesse);
        ~Hashtabelle_Device();
        
        size_t getzahlZellen();
        size_t getGroesseHashtabelle();
        Zelle<T1,T2> * getHashtabelle(); 
    
        hashtyp getHashTyp();
        
        void drucken();

        void insert_List(T1 * pSchluesselListe, T2 * pWerteListe, size_t pGroesse);
        void suchen_List(T1 * pSchluesselListe, size_t pGroesse);
};

#endif