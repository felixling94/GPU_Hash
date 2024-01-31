#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../include/statisch_hashtabelle.h>

template <typename T1, typename T2>
Statisch_Hashtabelle<T1,T2>::Statisch_Hashtabelle():Hashtabelle<T1>::Hashtabelle(){
    hashtyp_kode = keine_aufloesung;
    hashtabelle = new Zelle<T1,T2>[2];
};

template <typename T1, typename T2>
Statisch_Hashtabelle<T1,T2>::Statisch_Hashtabelle(hashtyp pHashtyp, size_t pGroesse):
Hashtabelle<T1>::Hashtabelle(pGroesse){
    hashtyp_kode = pHashtyp;
    hashtabelle = new Zelle<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Statisch_Hashtabelle<T1,T2>::~Statisch_Hashtabelle(){
};

template <typename T1, typename T2>
Zelle<T1,T2> * Statisch_Hashtabelle<T1,T2>::getHashtabelle(){
    return hashtabelle;
};

template <typename T1, typename T2>
std::string Statisch_Hashtabelle<T1,T2>::getZelle(size_t pIndex){
    std::string zeichenkette;

    if (pIndex < (this->groesseHashtabelle)){
        zeichenkette.append(std::to_string(hashtabelle[pIndex].schluessel));
        zeichenkette.append("  ");
        zeichenkette.append(std::to_string(hashtabelle[pIndex].wert));
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};

template <typename T1, typename T2>
void Statisch_Hashtabelle<T1,T2>::drucken(){
    if ((this->groesseHashtabelle) > 0){
        std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
        for(size_t i = 0; i < (this->groesseHashtabelle); i++)
            std::cout << i << "  " << getZelle(i) << std::endl;
    }else{
        std::cout << "Es gibt weder Schlüssel noch Werte in der Hashtabelle." << std::endl;
    }
};

//Berechne den Modulo eines Schlüssels
template <typename T1, typename T2>
size_t Statisch_Hashtabelle<T1,T2>::getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
};

template <typename T1, typename T2>
void Statisch_Hashtabelle<T1,T2>::insert(T1 pSchluessel, T2 pWert){
    size_t index_neu = this->getHashwert(pSchluessel);

    //Ohne Kollisionsauflösung
    if (hashtyp_kode == keine_aufloesung){
        if(hashtabelle[index_neu].schluessel == NULL){
            hashtabelle[index_neu] = new Zelle<T1,T2>{pSchluessel,pWert};
            return;
        }
        
    //Lineare Hashverfahren
    }else if(hashtyp_kode == linear_aufloesung){
        size_t i = 0;

        while(i < (this->groesseHashtabelle)){
            index_neu = this->getHashwert(index_neu + i);
            
            if(hashtabelle[index_neu] == NULL){
                hashtabelle[index_neu] = new Zelle<T1,T2>{pSchluessel,pWert};
                return;
            }

            ++i;
        }

    //Quadratische Hashverfahren
    }else if(hashtyp_kode== quadratisch_aufloesung){
        size_t i = 0;

        while((i/2) < (this->groesseHashtabelle)){
            index_neu = this->getHashwert(index_neu + getQuadratisch_Sondierungswert(i));
            
            if(hashtabelle[index_neu] == 0){
                hashtabelle[index_neu] = new Zelle<T1,T2>{pSchluessel,pWert};
                return;
            }

            ++i;
        }

    //Beliebige Hashverfahren
    }else{
        //TODO
    }   

    std::cout << "Der Schlüssel " << pSchluessel << " mit dem Index " << index_neu << " kann der Hashtabelle ";
    if (hashtyp_kode == keine_aufloesung){
        std::cout << "ohne Kollisionsauflösung ";
    }else if(hashtyp_kode == linear_aufloesung){
        std::cout << "mit linearem Sondieren ";
    }else if(hashtyp_kode == quadratisch_aufloesung){
        std::cout << "mit quadratischem Sondieren ";
    }else{
        std::cout << "mit beliebigen Hashverfahren ";
    }
    std::cout << "nicht hinzugefügt werden." << std::endl;

    return;
};


template <typename T1, typename T2>
bool Statisch_Hashtabelle<T1,T2>::suchen(T1 pSchluessel){
    size_t index_neu = this->getHashwert(pSchluessel);

    //Ohne Kollisionsauflösung
    if (hashtyp_kode == keine_aufloesung){
        if(hashtabelle[index_neu].schluessel = pSchluessel){
            return true;
        }else{
            return false;
        }

    //Lineare Hashverfahren
    }else if(hashtyp_kode == linear_aufloesung){
        size_t i = 0;
       
        while(i < (this->groesseHashtabelle)){
            index_neu = this->getHashwert(index_neu + i);
            
            if(hashtabelle[index_neu].schluessel == pSchluessel) return true;

            ++i;
        }
        
        return false;

    //Quadratische Hashverfahren
    }else if(hashtyp_kode == quadratisch_aufloesung){
        size_t i = 0;

        while ((i/2) < (this->groesseHashtabelle)){
            index_neu = this->getHashwert(index_neu + getQuadratisch_Sondierungswert(i));
            
            if (hashtabelle[index_neu].schluessel == pSchluessel) return true;
      
            ++i;
        }

        return false;

    //Beliebige Hashverfahren
    }else{
        //TODO
        return false;
    }   
};