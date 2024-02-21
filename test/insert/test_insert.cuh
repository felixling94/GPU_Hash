#ifndef TEST_INSERT_CUH
#define TEST_INSERT_CUH

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <../data/datenvorlage.h>
#include <../include/hashtabelle.h>
#include <../core/hashtabelle.cpp>

#include <../include/hashtabelle_device.cuh>
#include <../core/hashtabelle.cuh>
//#include <../templates/hashtabelle_uint32.cuh>

#include <../tools/timer.h>

template <typename T1, typename T2>
class Test_Insert{
    private:
        std::vector<T1> schluesselListe;
        std::vector<T2> werteListe;

        //Größe einer Zelle
        size_t NX = 800;
        size_t NY = 200;
        size_t NZ = 200;

        //Größe einer Hashtabelle
        size_t groesseHashtabelle = NX*NY*NZ + 5;

    public:
        //Konstruktor
        Test_Insert(){
            schluesselListe.reserve(NX*NY*NZ);
            werteListe.reserve(NX*NY*NZ + 5);
        };
        
        Test_Insert(size_t nx, size_t ny, size_t nz, size_t groesse_hashtabelle){
            if (nx*ny*nz>groesse_hashtabelle){
                std::cout << "Die Hashtabelle darf höchstens maximal " << groesse_hashtabelle;
                std::cout << " Datenelemente enthalten." << std::endl;
                std::cout << "Leider beträgt die Zahl der Zellen " << nx*ny*nz << "." << std::endl;
                exit (EXIT_FAILURE);
            }
            
            NX=nx;
            NY=ny;
            NZ=nz;
            groesseHashtabelle=groesse_hashtabelle;
            schluesselListe.reserve(nx*ny*nz);
            werteListe.reserve(groesse_hashtabelle);
        };

        //Destruktor
        ~Test_Insert(){};
        
        std::vector<T1> getSchluessel(){
            return schluesselListe;
        };

        std::vector<T2> getWerte(){
            return werteListe;
        };

        //Erzeuge verschiedene Werte für die Schlüssel und Werte zufällig
        void erzeuge_zellen(int min=0, int max=100){
            std::vector<T1> schluessel_vector;
            std::vector<T2> werte_vector;
            schluessel_vector.reserve(NX*NY*NZ);
            werte_vector.reserve(NX*NY*NZ);

            std::random_device zufallCPU;
            size_t seed = zufallCPU();
            std::mt19937 rnd(seed);
  
            std::uniform_int_distribution<T1> dist1(1,max);
            std::uniform_int_distribution<T2> dist2(min,max);

            for (size_t i = 0; i < NX*NY*NZ; i++){
                T1 rand1 = dist1(rnd);
                T2 rand2 = dist2(rnd);

                schluessel_vector.push_back(rand1);
                werte_vector.push_back(rand2);
            }

            std::copy(schluessel_vector.begin(),schluessel_vector.end(),schluesselListe.begin());
            std::copy(werte_vector.begin(),werte_vector.end(),werteListe.begin());
        };
        
        //Fuege der Hashtabelle eine Liste von Paaren von Schlüsseln und Werten hinzu       
        void insert_hashtabelle(hashtyp pHashtyp){
            //Deklariere und initialisiere alle Variablen
            const size_t nx{NX}, ny{NY}, nz{NZ};
            //bool equalArray;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Sequentielle Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////

            Zeit::starte();

            Hashtabelle<T1,T2> hashtabelle1(pHashtyp,groesseHashtabelle);
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            if (pHashtyp == keine_aufloesung){
                std::cout << "Ohne Kollisionauflösung" << std::endl;
            }else if(pHashtyp == linear_aufloesung){
                std::cout << "Lineare Hashverfahren" << std::endl;
            }else if(pHashtyp == quadratisch_aufloesung){
                std::cout << "Quadratische Hashverfahren" << std::endl;
            }else if(pHashtyp == doppelt_aufloesung){
                std::cout << "Doppelte Hashverfahren" << std::endl;
            }
            std::cout << "****************************************************************";
            std::cout << "***************" << std::endl;
            std::cout << "SEQUENTIELLE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;

            hashtabelle1.insert_List(schluesselListe.data(),werteListe.data(),nx,ny,nz);
            
            //Fasse Resultate für jede Hashverfahren zusammen
            Zeit::beende();
            
            std::cout << "Anzahl der Zellen in der Hashtabelle        : ";
            std::cout << hashtabelle1.getzahlZellen() << std::endl;
            std::cout << "Gesamtdauer (Millisekunden)                 : ";
            std::cout << Zeit::getDauer() << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////
            //Parallele Ausführung
            /////////////////////////////////////////////////////////////////////////////////////////
            Hashtabelle_Device<T1,T2> hashtabelle2(pHashtyp,groesseHashtabelle);
            std::cout << std::endl;
            std::cout << "PARALLELE AUSFÜHRUNG" << std::endl;
            std::cout << std::endl;
            hashtabelle2.insert_List(schluesselListe.data(),werteListe.data(),nx*ny*nz);
        };
};

#endif