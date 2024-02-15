#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../include/hashfunktionen.h>
#include <../include/hashtabelle.h>

template <typename T1, typename T2>
Hashtabelle<T1,T2>::Hashtabelle(){
    groesseHashtabelle = 2;
    hashtyp_kode = keine_aufloesung;
    hashfunktion_kode = modulo;
    hashtabelle = new Zelle<T1,T2>[2];
};

template <typename T1, typename T2>
Hashtabelle<T1,T2>::Hashtabelle(hashtyp pHashtyp, hashfunktion pHashfunktion, size_t pGroesse){
    groesseHashtabelle = pGroesse;
    hashtyp_kode = pHashtyp;
    hashfunktion_kode = pHashfunktion;
    hashtabelle = new Zelle<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Hashtabelle<T1,T2>::~Hashtabelle(){
};

//Berechne den Hashwert eines Schlüssels
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getHashwert(T1 pSchluessel){
    if (hashfunktion_kode == modulo){
        return Hashfunktionen::modulo_hash<T1>(pSchluessel,groesseHashtabelle);
    }else if (hashfunktion_kode == murmer){
        return Hashfunktionen::murmer_hash<T1>(pSchluessel,groesseHashtabelle);
    }
    return 0;
};

//Berechne den Wert einer Sondierungsfunktion eines Schlüssels
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
};

//Drucke die Zeile einer Hashtabelle
template <typename T1, typename T2>
std::string Hashtabelle<T1,T2>::getZelle(size_t pIndex){
    std::string zeichenkette;

    if (pIndex < (groesseHashtabelle)){
        if (hashtabelle[pIndex].schluessel !=  FeldLeer || hashtabelle[pIndex].wert !=  FeldLeer){
            zeichenkette.append(std::to_string(hashtabelle[pIndex].schluessel));
            zeichenkette.append("  ");
            zeichenkette.append(std::to_string(hashtabelle[pIndex].wert));
        }else{
            zeichenkette.append("Leer  Leer");
        } 
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};

//Gebe die Größe der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getGroesseHashtabelle(){
    return groesseHashtabelle;
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T1, typename T2>
hashtyp Hashtabelle<T1,T2>::getHashTyp(){
    return hashtyp_kode;
};

//Gebe den Typ der Hashfunktion einer Hashtabelle zurück
template <typename T1, typename T2>
hashfunktion Hashtabelle<T1,T2>::getHashfunktion(){
    return hashfunktion_kode;
};

//Gebe die Hashtabelle zurück
template <typename T1, typename T2>
Zelle<T1,T2> * Hashtabelle<T1,T2>::getHashtabelle(){
    return hashtabelle;
};

//Drucke die Hashtabelle
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::drucken(){
    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
    for(size_t i = 0; i < (groesseHashtabelle); i++) std::cout << i << "  " << getZelle(i) << std::endl;  
};

//Fuege der Hashtabelle ein Datenelement in der Hashtabelle hinzu
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::insert(T1 pSchluessel, T2 pWert){
    size_t index_neu = getHashwert(pSchluessel);

    //Ohne Kollisionsauflösung
    if (hashtyp_kode == keine_aufloesung){
        if(hashtabelle[index_neu].schluessel ==  FeldLeer || hashtabelle[index_neu].wert ==  FeldLeer){
            hashtabelle[index_neu].schluessel = pSchluessel;
            hashtabelle[index_neu].wert = pWert;
            return;
        }

    //Lineare Hashverfahren
    }else if(hashtyp_kode == linear_aufloesung){
        size_t i = 0;

        while(i < groesseHashtabelle){
            index_neu = (index_neu + i )%groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel ==  FeldLeer || hashtabelle[index_neu].wert ==  FeldLeer){
                hashtabelle[index_neu].schluessel = pSchluessel;
                hashtabelle[index_neu].wert = pWert;
                return;
            }

            ++i;
        }

    //Quadratische Hashverfahren
    }else if(hashtyp_kode== quadratisch_aufloesung){
        size_t i = 0;

        while((i/2) < groesseHashtabelle){
            index_neu = (index_neu+getQuadratisch_Sondierungswert(i)) % groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel ==  FeldLeer || hashtabelle[index_neu].wert ==  FeldLeer){            
                hashtabelle[index_neu].schluessel = pSchluessel;
                hashtabelle[index_neu].wert = pWert;
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

//Suche nach einem Schlüssel in der Hashtabelle
template <typename T1, typename T2>
bool Hashtabelle<T1,T2>::suchen(T1 pSchluessel){
    size_t index_neu;

    index_neu = getHashwert(pSchluessel);

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
       
        while(i < groesseHashtabelle){
            index_neu = (index_neu+i)%groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel == pSchluessel) return true;

            ++i;
        }
        
        return false;

    //Quadratische Hashverfahren
    }else if(hashtyp_kode == quadratisch_aufloesung){
        size_t i = 0;

        while ((i/2) < groesseHashtabelle){
           index_neu = (index_neu+getQuadratisch_Sondierungswert(i)) % groesseHashtabelle;
            
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