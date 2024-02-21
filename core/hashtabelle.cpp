#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../data/datenvorlage.h>
#include <../include/hashtabelle.h>
#include <../hashfunktionen/dycuckoo_funktionen.h>

#define PRIME_uint 294967291u

template <typename T1, typename T2>
Hashtabelle<T1,T2>::Hashtabelle():
hashtyp_kode(keine_aufloesung),groesseHashtabelle(2){
    hashtabelle = new Zelle<T1,T2>[2];
};

template <typename T1, typename T2>
Hashtabelle<T1,T2>::Hashtabelle(hashtyp pHashtyp, size_t pGroesse):
hashtyp_kode(pHashtyp),groesseHashtabelle(pGroesse){
    hashtabelle = new Zelle<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Hashtabelle<T1,T2>::~Hashtabelle(){
    delete[] hashtabelle;
};

//Berechne den Hashwert eines Schlüssels mithilfe von DyCuckoo-Hash3
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getHashwert(T1 pSchluessel){
    return DyCuckoo_Funktionen::hash3<T1>(pSchluessel)%groesseHashtabelle;
};

//Berechne den 2. Hashwert eines Schlüssels mithilfe von DyCuckoo-Hash35
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getHashwert2(T1 pSchluessel){
    return DyCuckoo_Funktionen::hash5<T1>(pSchluessel)%groesseHashtabelle;
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
        if (hashtabelle[pIndex].schluessel!= 0){
            zeichenkette.append(std::to_string(hashtabelle[pIndex].schluessel));
            zeichenkette.append("  ");
            zeichenkette.append(std::to_string(hashtabelle[pIndex].wert));
        }else{
            zeichenkette.append("Leer      Leer");
        } 
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};
//Gebe die Anzahl der Zellen in der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getzahlZellen(){
    size_t zahl = 0;
    for (size_t i=0; i<groesseHashtabelle; i++) if(hashtabelle[i].schluessel!=LeerFeld) ++zahl;
    return zahl;
};

//Gebe die Größe der Hashtabelle zurück
template <typename T1, typename T2>
size_t Hashtabelle<T1,T2>::getGroesseHashtabelle(){
    return groesseHashtabelle;
};

//Gebe die Hashtabelle zurück
template <typename T1, typename T2>
Zelle<T1,T2> * Hashtabelle<T1,T2>::getHashtabelle(){
    return hashtabelle;
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T1, typename T2>
hashtyp Hashtabelle<T1,T2>::getHashTyp(){
    return hashtyp_kode;
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
        if(hashtabelle[index_neu].schluessel==LeerFeld || hashtabelle[index_neu].schluessel==pSchluessel){
            hashtabelle[index_neu].schluessel = pSchluessel;
            hashtabelle[index_neu].wert = pWert;
        }

    //Lineare Hashverfahren
    }else if(hashtyp_kode == linear_aufloesung){
        size_t i, max_groesseHashtabelle;
        i = 0;
        max_groesseHashtabelle = (size_t)((100+PROZENT_SCHLEIFE)/100*groesseHashtabelle);

        while(i<max_groesseHashtabelle){
            index_neu=(index_neu+i)%groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel==LeerFeld || hashtabelle[index_neu].schluessel==pSchluessel){
                hashtabelle[index_neu].schluessel = pSchluessel;
                hashtabelle[index_neu].wert = pWert;
                break;
            }

            ++i;
        }

    //Quadratische Hashverfahren
    }else if(hashtyp_kode== quadratisch_aufloesung){
        size_t i, max_groesseHashtabelle;
        i = 0;
        max_groesseHashtabelle = (size_t)((100+PROZENT_SCHLEIFE)/100*groesseHashtabelle);

        while((i/2)<max_groesseHashtabelle){
            index_neu = (index_neu+getQuadratisch_Sondierungswert(i)) % groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel==LeerFeld || hashtabelle[index_neu].schluessel==pSchluessel){            
                hashtabelle[index_neu].schluessel = pSchluessel;
                hashtabelle[index_neu].wert = pWert;
                break;
            }

            ++i;
        }

    //Doppelte Hashverfahren
    }else{
        size_t i, max_groesseHashtabelle;
        i = 0;
        max_groesseHashtabelle = (size_t)((100+PROZENT_SCHLEIFE)/100*groesseHashtabelle);

        while(i<max_groesseHashtabelle){
            index_neu = (index_neu + i*getHashwert2(pSchluessel)) % groesseHashtabelle;
            
            if(hashtabelle[index_neu].schluessel==LeerFeld || hashtabelle[index_neu].schluessel==pSchluessel){            
                hashtabelle[index_neu].schluessel = pSchluessel;
                hashtabelle[index_neu].wert = pWert;
                break;
            }

            ++i;
        }
    }   
};

//Fuege der Hashtabelle ein Array von Schlüsseln und deren Werten gleichzeitig hinzu.
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::insert_List(T1 * pSchluesselListe, T2 * pWerteListe, const size_t nx, const size_t ny, const size_t nz){
    size_t i;
    
    for (size_t ix = 0; ix < nx; ++ix) {
        for (size_t iy = 0; iy < ny; ++iy) {
            for (size_t iz = 0; iz < nz; ++iz) {
                i = iz * (nz * ny) + iy * ny + ix;
                insert(pSchluesselListe[i], pWerteListe[i]);
            }
        }
    }
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

    //Doppelte Hashverfahren
    }else{
        //TODO
        return false;
    }   
};

//Suche nach einem Array von Schlüsseln in der Hashtabelle gleichzeitig.
template <typename T1, typename T2>
void Hashtabelle<T1,T2>::suchen_List(T1 * pSchluesselListe, const size_t nx, const size_t ny, const size_t nz){
    size_t i;
    
    for (size_t ix = 0; ix < nx; ++ix) {
        for (size_t iy = 0; iy < ny; ++iy) {
            for (size_t iz = 0; iz < nz; ++iz) {
                i = iz * (nz * ny) + iy * ny + ix;
                suchen(pSchluesselListe[i]);
            }
        }
    }
};