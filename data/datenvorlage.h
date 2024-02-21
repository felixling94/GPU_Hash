#ifndef DATENVORLAGE_H
#define DATENVORLAGE_H

#define PROZENT_SCHLEIFE 15

#define LeerFeld 0

//Arten der offenen Hashverfahren
//0: Keine Kollisionsauflösung
//1: Lineare Hashverfahren
//2: Quadratische Hashverfahren
//3: Doppelte Hashverfahren
enum hashtyp{keine_aufloesung=0, linear_aufloesung, 
             quadratisch_aufloesung, doppelt_aufloesung};

enum hashfunktion{modulo=0, murmer, multiplikativ};

template <typename T1, typename T2>
struct Zelle{
    T1 schluessel = LeerFeld;
    T2 wert;
};

template <typename T1, typename T2>
struct cuckoo_hashtabelle{
    Zelle<T1,T2> * hashtabelle;
    hashtyp hashtyp_kode;
    hashfunktion hashfunktion_kode;
};

#endif