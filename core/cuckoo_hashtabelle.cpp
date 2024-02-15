#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include <../include/hashtabelle.h>
#include <../include/cuckoo_hashtabelle.h>
#include <../include/hashfunktionen.h>

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::Cuckoo_Hashtabelle(){
    groesseHashtabelle = 2;

    hashtyp_kode1 = keine_aufloesung;
    hashfunktion_kode1 = modulo;
    hashtabelle1 = new Zelle<T1,T2>[2];

    hashtyp_kode2 = keine_aufloesung;
    hashfunktion_kode2 = modulo;
    hashtabelle2 = new Zelle<T1,T2>[2];
};

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::Cuckoo_Hashtabelle(hashtyp pHashtyp1, hashfunktion pHashfunktion1, hashtyp pHashtyp2, hashfunktion pHashfunktion2, size_t pGroesse){
    groesseHashtabelle = pGroesse;

    hashtyp_kode1 = pHashtyp1;
    hashfunktion_kode1 = pHashfunktion1;
    hashtabelle1 = new Zelle<T1,T2>[pGroesse];

    hashtyp_kode2 = pHashtyp2;
    hashfunktion_kode2 = pHashfunktion2;
    hashtabelle2 = new Zelle<T1,T2>[pGroesse];
};

template <typename T1, typename T2>
Cuckoo_Hashtabelle<T1,T2>::~Cuckoo_Hashtabelle(){
};

//Bestimme den Hashfunktion einer Hashtabelle
template <typename T1, typename T2>
size_t Cuckoo_Hashtabelle<T1,T2>::getHashwert(T1 pSchluessel, hashtabelle_nr pHashtabelle_nr){
    hashfunktion hashfunktion_kode_neu;

    if (pHashtabelle_nr == hashtabelle_1){
        hashfunktion_kode_neu = hashfunktion_kode1;
        return getHashfunktion(pSchluessel, hashfunktion_kode_neu);
    }else{
        hashfunktion_kode_neu = hashfunktion_kode2;
        hashfunktion_kode_neu = hashfunktion_kode1;
        return getHashfunktion(pSchluessel, hashfunktion_kode_neu);
    }
};

//Berechne den Hashwert eines Schlüssels
template <typename T1, typename T2>
size_t Cuckoo_Hashtabelle<T1,T2>:: getHashfunktion(T1 pSchluessel, hashfunktion pHashfunktion){
    if (pHashfunktion == modulo){
        return Hashfunktionen::modulo_hash<T1>(pSchluessel,groesseHashtabelle);
    }else if (pHashfunktion == murmer){
        return Hashfunktionen::murmer_hash<T1>(pSchluessel,groesseHashtabelle);
    }
    return 0;
};

//Berechne den Wert einer Sondierungsfunktion eines Schlüssels
template <typename T1, typename T2>
size_t Cuckoo_Hashtabelle<T1,T2>::getQuadratisch_Sondierungswert(size_t pIndex){
    size_t i = (size_t) pow(ceil((double)pIndex/2),2.0);
    size_t j = (size_t) pow(-1.0,(double)pIndex);
    return (i * j);
};

//Drucke die Zeile einer Hashtabelle
template <typename T1, typename T2>
std::string Cuckoo_Hashtabelle<T1,T2>::getZelle(size_t pIndex, hashtabelle_nr pHashtabelle_nr){
    std::string zeichenkette;

    if (pIndex < (groesseHashtabelle)){
        if (pHashtabelle_nr == hashtabelle_1){
            if (hashtabelle1[pIndex].schluessel !=  FeldLeer || hashtabelle1[pIndex].wert !=  FeldLeer){
                zeichenkette.append(std::to_string(hashtabelle1[pIndex].schluessel));
                zeichenkette.append("  ");
                zeichenkette.append(std::to_string(hashtabelle1[pIndex].wert));
            }else{
                zeichenkette.append("Leer  Leer");
            } 

        }else{
            if (hashtabelle2[pIndex].schluessel !=  FeldLeer || hashtabelle2[pIndex].wert !=  FeldLeer){
                zeichenkette.append(std::to_string(hashtabelle2[pIndex].schluessel));
                zeichenkette.append("  ");
                zeichenkette.append(std::to_string(hashtabelle2[pIndex].wert));
            }else{
                zeichenkette.append("Leer  Leer");
            } 
        }
    }else{
        zeichenkette.append("Der Index muss mindestens 0 und weniger als die Größe der Hashtabelle sein.");
    }

    return zeichenkette;
};

//Vertausche zwei Zellen in einer Hashtabelle
template <typename T1, typename T2>
Zelle<T1,T2> Cuckoo_Hashtabelle<T1,T2>::vertauschen(hashtabelle_nr pHashtabelle_nr, size_t pIndex, Zelle<T1,T2> zelle_eingabe){
    Zelle<T1,T2> zelle_temp;

    if (pHashtabelle_nr == hashtabelle_1){
        zelle_temp.schluessel = hashtabelle1[pIndex].schluessel;
        zelle_temp.wert = hashtabelle1[pIndex].wert;

        hashtabelle1[pIndex].schluessel = zelle_eingabe.schluessel;
        hashtabelle1[pIndex].wert = zelle_eingabe.wert;

        return zelle_temp;
    }else{
        zelle_temp.schluessel = hashtabelle2[pIndex].schluessel;
        zelle_temp.wert = hashtabelle2[pIndex].wert;

        hashtabelle2[pIndex].schluessel = zelle_eingabe.schluessel;
        hashtabelle2[pIndex].wert = zelle_eingabe.wert;

        return zelle_temp;
    }
};


//Gebe die Größe der Hashtabelle zurück
template <typename T1, typename T2>
size_t Cuckoo_Hashtabelle<T1,T2>::getGroesseHashtabelle(){
    return groesseHashtabelle;
};

//Gebe den Hashtyp einer Hashtabelle zurück
template <typename T1, typename T2>
hashtyp Cuckoo_Hashtabelle<T1,T2>::getHashTyp(hashtabelle_nr pHashtabelle_nr){
    if (pHashtabelle_nr == hashtabelle_1){
        return hashtyp_kode1;
    }else{
        return hashtyp_kode2;
    }
};

//Gebe die Hashtabelle zurück
template <typename T1, typename T2>
Zelle<T1,T2> * Cuckoo_Hashtabelle<T1,T2>::getHashtabelle(hashtabelle_nr pHashtabelle_nr){
    if (pHashtabelle_nr == hashtabelle_1){
        return hashtabelle1;
    }else{
        return hashtabelle2;
    }
}; 

//Drucke die Hashtabelle
template <typename T1, typename T2>
void Cuckoo_Hashtabelle<T1,T2>::drucken(){
    std::cout << "1. Hashtabelle" << std::endl;
    std::cout << std::endl;
    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
    for(size_t i = 0; i < (groesseHashtabelle); i++) std::cout << i << "  " << getZelle(i,hashtabelle_1) << std::endl;  
    std::cout << std::endl;

    std::cout << "2. Hashtabelle" << std::endl;
    std::cout << std::endl;
    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
    for(size_t i = 0; i < (groesseHashtabelle); i++) std::cout << i << "  " << getZelle(i,hashtabelle_2) << std::endl;  
    std::cout << std::endl;
};

//Fuege eine der zwei Hashtabellen ein Datenelement in einer der zwei Hashtabelle hinzu
template <typename T1, typename T2>
void Cuckoo_Hashtabelle<T1,T2>::insert(T1 pSchluessel, T2 pWert){
    Zelle<T1,T2> zellen_temp;
    size_t i, index_neu1, index_neu2;

    i = 1;
    zellen_temp.schluessel = pSchluessel;
    zellen_temp.wert = pWert;
    index_neu1 = getHashwert(pSchluessel,hashtabelle_1);
    index_neu2 = getHashwert(pSchluessel,hashtabelle_2);
 
    if (hashtabelle1[index_neu1].schluessel == pSchluessel || hashtabelle2[index_neu2].schluessel == pSchluessel) return;
    
    while (i<groesseHashtabelle){
        index_neu1 = (index_neu1 + i )%groesseHashtabelle;
        zellen_temp = vertauschen(hashtabelle_1,index_neu1,zellen_temp);

        if (zellen_temp.schluessel == FeldLeer) return;

        index_neu2 = (index_neu2 + i )%groesseHashtabelle; 
        zellen_temp = vertauschen(hashtabelle_2,index_neu2,zellen_temp); 

        if (zellen_temp.schluessel == FeldLeer) return;
        
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