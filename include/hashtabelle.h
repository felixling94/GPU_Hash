#ifndef HASHTABELLE_H
#define HASHTABELLE_H

//Arten der offenen Hashverfahren
//0: Keine Kollisionsauflösung
//1: Lineare Hashverfahren
//2: Quadratische Hashverfahren
enum hashtyp{keine_aufloesung=0, linear_aufloesung,
            quadratisch_aufloesung, cuckoo_aufloesung};

template <typename T1, typename T2>
struct Knoten{
    T1 schluessel;
    T2 wert;
};

template <typename T0, typename T1, typename T2>
struct Kollision{
    Knoten<T1,T2> knoten;
    T0 zahlKollision = 0;
};

template <typename T0, typename T1, typename T2>
class Hashtabelle{
    protected:
        hashtyp hashtyp_kode;
        bool standDruecke;

        Knoten<T1,T2> * hashtabelle;
        T0 groesseHashtabelle;

    public:
        Hashtabelle();
        Hashtabelle(hashtyp pHashtyp, T0 pGroesse);
        ~Hashtabelle();
        
        T0 getGroesseHashtabelle();
        Knoten<T1,T2> * getHashtabelle();
        bool getStandDruecke();

        void schalteStandDruecke();
        void druckeHashtabelle();
        void druckeHashtabelle1(Knoten<T1,T2> * pHashtabelle);

        T0 hashwert(T1 pSchluessel);
        void insert(Knoten<T1,T2> pKnoten);
        void insert_CUDA(Knoten<T1,T2> * pKnoten, size_t pKnotenGroesse, Kollision<T0,T1,T2> * pKollision);
        bool suchen(Knoten<T1,T2> pKnoten);
        void suchen_CUDA(Knoten<T1,T2> * pKnoten, size_t pKnotenGroesse, Kollision<T0,T1,T2> * pKollision);
};

#endif