#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <cmath>

#include <../include/dynamisch_hashtabelle.h>

template <typename T1, typename T2>
Dynamisch_Hashtabelle<T1,T2>::Dynamisch_Hashtabelle():Hashtabelle<T1>::Hashtabelle(){
    standAuflosung = false;
    hashtabelle = new Liste<T1,T2>[2];
};

template <typename T1, typename T2>
Dynamisch_Hashtabelle<T1,T2>::Dynamisch_Hashtabelle(bool pStandAuflosung, size_t pGroesse):
Hashtabelle<T1>::Hashtabelle(pGroesse){
    standAuflosung = pStandAuflosung;
    hashtabelle = new Liste<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Dynamisch_Hashtabelle<T1,T2>::~Dynamisch_Hashtabelle(){
};

template <typename T1, typename T2>
Liste<T1,T2> * Dynamisch_Hashtabelle<T1,T2>::getHashtabelle(){
    return hashtabelle;
};

template <typename T1, typename T2>
std::string Dynamisch_Hashtabelle<T1,T2>::getZelle(size_t pIndex){
    std::string zeichenkette;

    if (pIndex < (this->groesseHashtabelle)){
        zeichenkette.append(std::to_string(hashtabelle[pIndex].jetzt.schluessel));
        zeichenkette.append("  ");
        zeichenkette.append(std::to_string(hashtabelle[pIndex].jetzt.wert));
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};

template <typename T1, typename T2>
void Dynamisch_Hashtabelle<T1,T2>::drucken(){
    if ((this->groesseHashtabelle) > 0){
        std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
        for(size_t i = 0; i < (this->groesseHashtabelle); i++)
            std::cout << i << "  " << getZelle(i) << std::endl;
    }else{
        std::cout << "Es gibt weder Schlüssel noch Werte in der Hashtabelle." << std::endl;
    }
};

template <typename T1, typename T2>
void Dynamisch_Hashtabelle<T1,T2>::insert(T1 pSchluessel, T2 pWert){
    //TODO
};

template <typename T1, typename T2>
bool Dynamisch_Hashtabelle<T1,T2>::suchen(T1 pSchluessel){
    //TODO
    return true;
};