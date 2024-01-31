#ifndef HASHTABELLE_H
#define HASHTABELLE_H

template <typename T1, typename T2>
struct Zelle{
    T1 schluessel = 0;
    T2 wert = 0;
};

template <typename T1>
class Hashtabelle{
    protected:
        size_t groesseHashtabelle;

        size_t getHashwert(T1 pSchluessel);

    public:
        Hashtabelle();
        Hashtabelle(size_t pGroesse);
        ~Hashtabelle();
        
        size_t getGroesseHashtabelle();
};

#endif