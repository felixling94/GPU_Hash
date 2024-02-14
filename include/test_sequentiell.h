#ifndef TEST_SEQUENTIELL_H
#define TEST_SEQUENTIELL_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <../include/hashfunktionen.h>
#include <../include/hashtabelle.h>
#include <../core/hashtabelle.cpp>
#include <../core/cuckoo_hashtabelle.cpp>
#include <../tools/timer.h>

template <typename T1, typename T2>
class Test_Sequentiell{
    private:
        size_t schluessel_max;
        size_t werte_max;

        std::vector<T1> test_schluessel;
        std::vector<T2> test_werte;

        size_t zellen_zahl;
        size_t test_groesseHashtabelle;

    public:
        Test_Sequentiell():schluessel_max(20),werte_max(20),zellen_zahl(10),test_groesseHashtabelle(10){
            test_schluessel.reserve(10);
            test_werte.reserve(10);
        };

        Test_Sequentiell(size_t pMaxSchluessel, size_t pMaxWerte, size_t pZahlZellen, size_t pGroesseHashtabelle):
        schluessel_max(pMaxSchluessel),werte_max(pMaxWerte){
            if (pZahlZellen>pGroesseHashtabelle){
                std::cout << "Die Hashtabelle darf höchstens maximal " << pGroesseHashtabelle;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << pZahlZellen << "." << std::endl;
                exit (EXIT_FAILURE);
            }
            zellen_zahl = pZahlZellen;
            test_groesseHashtabelle = pGroesseHashtabelle;
            test_schluessel.reserve(pZahlZellen);
            test_werte.reserve(pZahlZellen);
        };

        ~Test_Sequentiell(){};

        std::vector<T1> getTest_schluessel(){
            return test_schluessel;
        };

        std::vector<T2> getTest_werte(){
            return test_werte;
        };
        
        //Erzeuge verschiedene Schlüssel zufällig
        void erzeuge_schluessel(){
            std::vector<T1> schluessel_vector;
            schluessel_vector.reserve(zellen_zahl);

            std::random_device zufallCPU;
            size_t seed = zufallCPU();
            std::mt19937 rnd(seed);
  
            std::uniform_int_distribution<T1> distribution(0, schluessel_max);

            for (size_t i = 0; i < zellen_zahl; i++){
                T1 rand;
                rand = distribution(rnd);
                schluessel_vector.push_back(rand);
            }

            std::copy(schluessel_vector.begin(),schluessel_vector.end(),test_schluessel.begin());
        };

        //Erzeuge verschiedene Werte zufällig
        void erzeuge_werte(){
            std::vector<T2> werte_vector;
            werte_vector.reserve(zellen_zahl);

            std::random_device zufallCPU;
            size_t seed = zufallCPU();
            std::mt19937 rnd(seed);
  
            std::uniform_int_distribution<T2> distribution(0, werte_max);

            for (size_t i = 0; i < zellen_zahl; i++){
                T2 rand;
                rand = distribution(rnd);
                werte_vector.push_back(rand);
            }

            std::copy(werte_vector.begin(),werte_vector.end(),test_werte.begin());
        };
        
        //Mische verschiedene Schlüssel
        void mische_schluessel(){
            std::vector<T1> schluessel_eingabe;
            schluessel_eingabe.reserve(zellen_zahl);

            std::random_device zufallCPU;
            size_t seed = zufallCPU();
            std::mt19937 rnd(seed);

            std::copy(test_schluessel.begin(), test_schluessel.end(),schluessel_eingabe.begin());
            std::shuffle(schluessel_eingabe.begin(), schluessel_eingabe.end(), rnd);

            std::copy(schluessel_eingabe.begin(),schluessel_eingabe.end(),test_schluessel.begin());
        };

        //Mische verschiedene Werte
        void mische_werte(){
            std::vector<T1> werte_eingabe;
            werte_eingabe.reserve(zellen_zahl);

            std::random_device zufallCPU;
            size_t seed = zufallCPU();
            std::mt19937 rnd(seed);

            std::copy(test_werte.begin(), test_werte.end(),werte_eingabe.begin());
            std::shuffle(werte_eingabe.begin(), werte_eingabe.end(), rnd);

            std::copy(werte_eingabe.begin(),werte_eingabe.end(),test_werte.begin());
        };

        void insert_hashtabelle(hashtyp pHashtyp, hashfunktion pHashfunktion, bool pStandDruecke){
            /////////////////////////////////////////////////////////////////////////////////////////
            //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten hinzu
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "Speicherung von " << zellen_zahl << " Zellen in der Hashtabelle mit der Größe von ";
            std::cout << test_groesseHashtabelle << " Datenelementen" << std::endl;
            std::cout << std::endl;
            
            T1 * hash_schluessel = test_schluessel.data();
            T2 * hash_werte = test_werte.data();

            Zeit::starte();
            
            //Erstelle eine ggf. statische Hashtabelle
            Hashtabelle<T1,T2> hashtabelle(pHashtyp,pHashfunktion,test_groesseHashtabelle);
            
            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            if (pHashtyp == keine_aufloesung){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(pHashtyp == linear_aufloesung){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(pHashtyp == quadratisch_aufloesung){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(pHashtyp == beliebig_aufloesung){
                std::cout << "Beliebige Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            
            for (size_t i=0; i<zellen_zahl; i++) hashtabelle.insert(hash_schluessel[i],hash_werte[i]);
            
            if (pStandDruecke == true) hashtabelle.drucken();
            
            //Fasse Resultate für jede Hashverfahren zusammen
            Zeit::beende();

            std::cout << std::endl;
            std::cout << "Gesamtdauer bei ";
            if (pHashtyp == keine_aufloesung){
                std::cout << "keiner Kollisionsauflösung: " << std::endl;
            }else if(pHashtyp == linear_aufloesung){
                std::cout << "linearen Hashverfahren: " << std::endl;
            }else if(pHashtyp == quadratisch_aufloesung){
                std::cout << "quadratischen Hashverfahren: " << std::endl;
            }else if(pHashtyp == beliebig_aufloesung){
                std::cout << "beliebigen Hashverfahren: " << std::endl;
            }
            std::cout << Zeit::getDauer() << " Millisekunden." << std::endl;
        };

        void insert_Cuckoo(hashtyp pHashtyp1, hashfunktion pHashfunktion1, hashtyp pHashtyp2, hashfunktion pHashfunktion2,bool pStandDruecke){
            /////////////////////////////////////////////////////////////////////////////////////////
            //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten hinzu
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "Speicherung von " << zellen_zahl << " Zellen in der Hashtabelle mit der Größe von ";
            std::cout << test_groesseHashtabelle << " Datenelementen" << std::endl;
            std::cout << std::endl;
            
            T1 * hash_schluessel = test_schluessel.data();
            T2 * hash_werte = test_werte.data();

            //Erstelle eine ggf. Cuckoo-Hashtabelle
            Cuckoo_Hashtabelle<T1,T2> cuckooHashtabelle(pHashtyp1, pHashfunktion1, pHashtyp2, pHashfunktion2, test_groesseHashtabelle);

            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            std::cout << "Cuckoo-Hashverfahren" << std::endl;
            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            
            for (size_t i=0; i<zellen_zahl; i++) cuckooHashtabelle.insert(hash_schluessel[i],hash_werte[i]);

            if (pStandDruecke == true) cuckooHashtabelle.drucken();
            
        };

        void suche_hashtabelle(hashtyp pHashtyp, hashfunktion pHashfunktion, bool pStandDruecke){
            /////////////////////////////////////////////////////////////////////////////////////////
            //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten hinzu
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "Speicherung von " << zellen_zahl << " Zellen in der Hashtabelle mit der Größe von ";
            std::cout << test_groesseHashtabelle << " Datenelementen" << std::endl;
            std::cout << std::endl;

            T1 * hash_schluessel = test_schluessel.data();
            T2 * hash_werte = test_werte.data();

            Zeit::starte();
            
            //Erstelle eine ggf. statische Hashtabelle
            Hashtabelle<T1,T2> hashtabelle(pHashtyp,pHashfunktion,test_groesseHashtabelle);
            
            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            if (pHashtyp == keine_aufloesung){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(pHashtyp == linear_aufloesung){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(pHashtyp == quadratisch_aufloesung){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(pHashtyp == beliebig_aufloesung){
                std::cout << "Beliebige Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "****************************************************************" << std::endl;
            
            for (size_t i=0; i<zellen_zahl; i++) hashtabelle.insert(hash_schluessel[i],hash_werte[i]);
            
            if (pStandDruecke == true) hashtabelle.drucken();

            //Fasse Resultate für jede Hashverfahren zusammen
            Zeit::beende();
            std::cout << std::endl;
            std::cout << "Gesamtdauer (in Millisekunden)                                    : ";
            std::cout << Zeit::getDauer() << std::endl;
            std::cout << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Suche nach einer Liste von Schlüsseln in der Hashtabelle
            /////////////////////////////////////////////////////////////////////////////////////////
            std::cout << "Suche nach " << zellen_zahl << " Zellen in der Hashtabelle mit der Größe von ";
            std::cout << test_groesseHashtabelle << " Datenelementen"<< std::endl;
            std::cout << std::endl;

            mische_schluessel();
            T1 * schluesselListe = test_schluessel.data();
            
            Zeit::starte();
                     
            for (size_t i=0; i<zellen_zahl; i++) hashtabelle.suchen(schluesselListe[i]);
            
            if (pStandDruecke == true) hashtabelle.drucken();
            
            //Fasse Resultate für jede Hashverfahren zusammen
            Zeit::beende();
            std::cout << std::endl;
            std::cout << "Gesamtdauer (in Millisekunden)                                    : ";
            std::cout << Zeit::getDauer() << std::endl;
            std::cout << std::endl;
        };
};

#endif