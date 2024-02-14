#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../include/hashtabelle.h>
#include <../include/cuckoo_hashtabelle.h>
#include <../include/hashfunktionen.h>

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::Cuckoo_Hashtabelle(){
    Hashtabelle<T1,T2> pHashtabelle1;
    Hashtabelle<T1,T2> pHashtabelle2;

    hashtabelle1 = pHashtabelle1;
    hashtabelle2 = pHashtabelle2;
};

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::Cuckoo_Hashtabelle(hashtyp pHashtyp1, hashfunktion pHashfunktion1, hashtyp pHashtyp2, hashfunktion pHashfunktion2, size_t pGroesse){
    Hashtabelle<T1,T2> pHashtabelle1(pHashtyp1,pHashfunktion1,pGroesse);
    Hashtabelle<T1,T2> pHashtabelle2(pHashtyp2,pHashfunktion2,pGroesse);

    hashtabelle1 = pHashtabelle1;
    hashtabelle2 = pHashtabelle2;
};

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::~Cuckoo_Hashtabelle(){
};

//Berechne den Hashwert eines Schlüssels
template <typename T1, typename T2>
size_t Cuckoo_Hashtabelle<T1,T2>::getHashwert(T1 pSchluessel, size_t pGroesseHashtabelle, hashfunktion pHashfunktion){
    if (pHashfunktion== modulo){
        return Hashfunktionen::modulo_hash<T1>(pSchluessel,pGroesseHashtabelle);
    }else if (pHashfunktion == murmer){
        return Hashfunktionen::murmer_hash<T1>(pSchluessel,pGroesseHashtabelle);
    }
    return 0;
};

//Drucke die Hashtabelle
template <typename T1, typename T2>
void Cuckoo_Hashtabelle<T1,T2>::drucken(){
    hashtabelle1.drucken();
    std::cout << std::endl;
    hashtabelle2.drucken();
};

//Fuege eine der zwei Hashtabellen ein Datenelement in einer der zwei Hashtabelle hinzu
template <typename T1, typename T2>
void Cuckoo_Hashtabelle<T1,T2>::insert(T1 pSchluessel, T2 pWert){
    Zelle<T1,T2> * hashtabelle_neu1;
    Zelle<T1,T2> * hashtabelle_neu2;
    Zelle<T1,T2> temp, zelle_neu1, zelle_neu2;

    size_t index_neu1, index_neu2;

    hashtabelle_neu1 = hashtabelle1.getHashtabelle();
    hashtabelle_neu2 = hashtabelle2.getHashtabelle();

    index_neu1 = getHashwert(pSchluessel,hashtabelle1.getGroesseHashtabelle(),hashtabelle1.getHashfunktion());
    index_neu2 = getHashwert(pSchluessel,hashtabelle2.getGroesseHashtabelle(),hashtabelle2.getHashfunktion());
 
    if (hashtabelle_neu1[index_neu1].schluessel == pSchluessel || hashtabelle_neu2[index_neu2].schluessel == pSchluessel){
        return;
    }

    size_t i, groesseHashtabelle;
    i = 1;
    groesseHashtabelle = hashtabelle1.getGroesseHashtabelle();

    while (i<groesseHashtabelle){
        index_neu1 = (index_neu1 + i )%groesseHashtabelle;
        
        temp.schluessel = hashtabelle_neu1[index_neu1].schluessel;
        temp.wert = hashtabelle_neu1[index_neu1].wert;

        hashtabelle_neu1[index_neu1].schluessel = pSchluessel;
        hashtabelle_neu1[index_neu1].wert = pWert;

        zelle_neu1 = temp;

        std::cout << "Hashtabelle: " << hashtabelle_neu1[index_neu1].schluessel << "  " << hashtabelle_neu1[index_neu1].wert << std::endl;
        std::cout << "Zelle: "<< zelle_neu1.schluessel << "  " << zelle_neu1.wert << std::endl;

        if (zelle_neu1.schluessel == FeldLeer) return;  

        index_neu2 = (index_neu2 + i )%groesseHashtabelle;
        
        temp.schluessel = hashtabelle_neu2[index_neu2].schluessel;
        temp.wert = hashtabelle_neu2[index_neu2].wert;

        hashtabelle_neu2[index_neu2].schluessel = pSchluessel;
        hashtabelle_neu2[index_neu2].wert = pWert;

        zelle_neu2 = temp;

        if (zelle_neu2.schluessel == FeldLeer) return;
        
        ++i;
    }
    
    std::cout << "Der Schlüssel " << pSchluessel << " kann der Hashtabelle ";
    std::cout << "mit Cuckoo-Hashverfahren nicht hinzugefügt werden." << std::endl;

    return;
};

//Suche nach einem Schlüssel in einer der zwei Hashtabellen
template <typename T1, typename T2>
bool Cuckoo_Hashtabelle<T1,T2>::suchen(T1 pSchluessel){
    return true;
    //TODO
};